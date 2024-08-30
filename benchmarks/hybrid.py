# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: hybrid.py
# ==============================================================================#

"""The Spectral State Space Model architecture with attention for benchmark tasks."""

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from models.stu.stu_utils_old import (
    get_top_eigh, 
    preconvolve,
    compute_ar_u,
    compute_spectral,
    compute_ar_y,
)
from models.transformer.attn_old import CausalSelfAttention
from utils.nearest_power_of_2 import nearest_power_of_2
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU


@dataclass
class SpectralHybridConfigs:
    d_in: int = 10
    d_out: int = 10
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 8
    sl: int = 1_000
    mlp_scale: int = 4
    embd_scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 16
    k_y: int = 2
    k_u: int = 3
    learnable_m_y: bool = True
    alpha: float = 0.9
    use_ar_y: bool = False
    use_ar_u: bool = False
    use_hankel_L: bool = False
    pct_attn: float = 0.50
    task: str = "copy"
    vocab_size: int = 20
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    device: torch.device = None

    # MoE
    moe: bool = True
    num_experts: int = 8
    num_experts_per_timestep: int = 2

    # Attention settings
    flash_attn: bool = True
    use_sq_relu: bool = False
    dilated_attn: bool = False
    segment_lengths: list[int] = field(default_factory=lambda: [128])
    dilated_ratios: list[int] = field(default_factory=lambda: [1])
    seq_parallel: bool = True
    xpos_rel_pos: bool = True
    xpos_scale_base: int = 512
    rms_norm_eps: float = 1e-5
    multiway: bool = False


class STU(nn.Module):
    """
    An STU (Spectral Transform Unit) layer.

    Args:
        configs: Configuration contains (at least) the following attributes:
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            num_eigh (int): Number of spectral filters to use.
            use_ar_y (bool): Use autoregressive on the output sequence?
            use_ar_u (bool): Use autoregressive on the input sequence?
            k_u (int): Autoregressive depth on the input sequence.
            k_y (int): Autoregressive depth on the output sequence.
            learnable_m_y (bool): Learn the M_y matrix?
            dropout (float): Dropout rate.
        sigma (torch.Tensor): Eigenvalues of the Hankel matrix.
        V (torch.Tensor): Precomputed FFT of top K eigenvectors.
    """

    def __init__(self, configs, sigma, V, padded_sl) -> None:
        super(STU, self).__init__()
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.d_model = configs.d_model
        self.embd_scale = configs.embd_scale
        self.k = configs.num_eigh
        self.use_ar_y = configs.use_ar_y
        self.use_ar_u = configs.use_ar_u
        self.sigma = sigma
        self.V = V  # Precomputed FFT of top K eigenvectors.
        self.padded_sl = padded_sl
        self.k_u = configs.k_u
        self.k_y = configs.k_y
        assert (self.k_u < self.k) or (self.k_y < self.k), "Cannot shift"
        self.learnable_m_y = configs.learnable_m_y
        self.stu_dropout = nn.Dropout(configs.dropout)
        self.resid_dropout = nn.Dropout(configs.dropout)

        # Parameterizable matrix Mᵘ, Mᵠ⁺, and Mᵠ⁻, per section 3
        self.M_u = nn.Parameter(torch.empty(self.k_u, self.d_model, self.d_model))
        self.M_phi_plus = nn.Parameter(torch.empty(self.k, self.d_model, self.d_model))
        self.M_phi_minus = nn.Parameter(torch.empty(self.k, self.d_model, self.d_model))

        # Parametrizable matrix Mʸ Introduced in section 5, equation 5
        if self.learnable_m_y:
            self.M_y = nn.Parameter(torch.zeros(self.d_model, self.k_y, self.d_model))
        else:
            self.register_buffer(
                "m_y", torch.zeros(self.d_model, self.k_y, self.d_model)
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STU layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (bsz, sl, d_out)
        """

        spectral = self.stu_dropout(
            compute_spectral(
                inputs,
                self.sigma,
                self.V,
                self.M_phi_plus,
                self.M_phi_minus,
                self.M_y,
                self.padded_sl,
                self.use_ar_y,
            )
        )

        match (self.use_ar_u, self.use_ar_y):
            case (True, True):
                y_t = spectral + compute_ar_u(self.M_u, inputs)
                y_t += compute_ar_y(self.M_y, y_t)
            case (True, False):
                y_t = spectral + compute_ar_u(self.M_u, inputs)
            case (False, True):
                y_t = spectral
                y_t += compute_ar_y(self.M_y, y_t)
            case (False, False):
                y_t = spectral

        return self.resid_dropout(y_t)


class MLP(nn.Module):
    """
    Multi-layer perceptron network using SwiGLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            mlp_scale (float): Scaling factor for hidden dimension.
            d_model (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs) -> None:
        super(MLP, self).__init__()
        self.swiglu = SwiGLU(
            dim=configs.d_model,
            h_dim=int(configs.mlp_scale * configs.d_model),
            bias=configs.bias,
            use_sq_relu=configs.use_sq_relu,
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.swiglu(x)
        x = self.dropout(x)
        return x


class GatedMLP(nn.Module):
    """
    Gated multi-layer perceptron network using SiLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            d_model (int): Input and output embedding dimension.
            mlp_scale (float): Scaling factor for hidden dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super().__init__()
        self.in_features = configs.d_model
        self.out_features = configs.d_model
        self.chunks = 2
        self.hidden_features = int(configs.mlp_scale * configs.d_model)

        self.fc_1 = nn.Linear(
            self.in_features, self.chunks * self.hidden_features, bias=configs.bias
        )
        self.fc_2 = nn.Linear(
            self.hidden_features, self.out_features, bias=configs.bias
        )
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GatedMLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        y = self.fc_1(x)
        y, gate = y.chunk(self.chunks, dim=-1)
        y = y * self.silu(gate)
        y = self.fc_2(y)
        return self.dropout(y)


class HybridBlock(nn.Module):
    """
    A single block of the hybrid spectral SSM model (spectral filtering + attention).

    Args:
        configs: Configuration object for STU and MLP layers
        sigma (torch.Tensor): Eigenvalues of the Hankel matrix.
        V (torch.Tensor): Precomputed FFT of top K eigenvectors.
        padded_sl (int): Padded sequence length for FFT operations.
    """

    def __init__(self, configs, sigma, V, padded_sl) -> None:
        super(HybridBlock, self).__init__()
        self.stu = STU(configs, sigma, V, padded_sl)
        self.attn = CausalSelfAttention(configs)

        self.mlp_1 = (
            MoE(
                configs,
                experts=[GatedMLP(configs) for _ in range(configs.num_experts)],
                gate=nn.Linear(configs.d_model, configs.num_experts, bias=configs.bias),
            )
            if configs.moe
            else GatedMLP(configs)
        )

        self.mlp_2 = (
            MoE(
                configs,
                experts=[GatedMLP(configs) for _ in range(configs.num_experts)],
                gate=nn.Linear(configs.d_model, configs.num_experts, bias=configs.bias),
            )
            if configs.moe
            else GatedMLP(configs)
        )

        self.rn_1 = RMSNorm(configs.d_model)
        self.rn_2 = RMSNorm(configs.d_model)
        self.rn_3 = RMSNorm(configs.d_model)
        self.rn_4 = RMSNorm(configs.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid spectral filtering + attention block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # STU portion
        z = x
        x = x + self.stu(self.rn_1(x))
        x = x + self.mlp_1(self.rn_2(x)) + z

        # Attention portion
        x = x + self.attn(self.rn_3(x))
        x = x + self.mlp_2(self.rn_4(x)) + z

        return x


class SpectralHybrid(nn.Module):
    """
    Model architecture based on stacked blocks of STU, attention, and MLP layers.

    Args:
        configs: Configuration object containing model hyperparameters.
    """

    def __init__(self, configs) -> None:
        super(SpectralHybrid, self).__init__()
        self.configs = configs
        self.n_layers = configs.n_layers
        self.d_model = configs.d_model
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.embd_scale = configs.embd_scale
        self.sl = configs.sl
        self.num_eigh = configs.num_eigh
        self.learnable_m_y = configs.learnable_m_y
        self.alpha = configs.alpha
        self.use_hankel_L = configs.use_hankel_L
        self.device = configs.device

        self.sigma, self.phi = get_top_eigh(
            self.sl, self.num_eigh, self.use_hankel_L, self.device
        )
        self.V, self.padded_sl = preconvolve(self.phi, self.sl)  # Precomputed.

        self.bias = configs.bias
        self.dropout = configs.dropout

        self.hybrid = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(self.sl, self.d_model),
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList(
                    [
                        HybridBlock(self.configs, self.sigma, self.V, self.padded_sl)
                        for _ in range(self.n_layers)
                    ]
                ),
            )
        )

        if configs.task == "adding":
            self.embed = nn.Linear(self.d_in, self.d_model)
        elif configs.task in ["copy", "induction"]:
            self.embed = nn.Embedding(self.d_in, self.d_model)

        self.output = nn.Linear(self.d_model, self.d_out, bias=configs.bias)
        self.loss_fn = self.configs.loss_fn
        self.task = self.configs.task

        # Initialize all weights
        self.m_x = self.d_out**-0.5
        self.std = (self.d_model) ** -0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print(
            "\nSTU-Attention Hybrid Model Parameter Count: %.2fM"
            % (self.get_num_params() / 1e6,)
        )

    def forward(self, inputs, targets):
        """
        Forward pass of the STU-Attention Hybrid model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor): Target tensor for loss computation

        Returns:
            Type (ignore due to high variability):
            - Predictions tensor
            - Tuple containing loss and metrics (if applicable)
        """
        if self.task in ["copy", "induction", "associative"]:
            x = self.embed(inputs)
        elif self.task == "adding":
            x = inputs.view(inputs.shape[0], -1, self.configs.d_in)
            x = self.embed(x)

        bsz, sl, _ = x.size()
        pos = torch.arange(0, sl, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.hybrid.wpe(pos)
        x = x + pos_emb

        x = self.hybrid.dropout(x)
        for block in self.hybrid.hidden:
            x = block(x)

        if self.task == "associative":
            logits = self.output(x[:, -1, :])
        else:
            logits = self.output(x)

        # Compute predictions
        if self.task in ["copy", "induction"]:
            preds = torch.argmax(logits, dim=-1)
        elif self.task == "adding":
            preds = logits.mean(dim=1).squeeze(-1)

        if targets is not None:
            if self.task == "copy":
                loss = self.loss_fn(logits.view(-1, self.d_out), targets.view(-1))
            elif self.task == "induction":
                logits_flat = logits.view(
                    -1, logits.size(-1)
                )  # Shape: (bsz * sl, vocab_size)
                targets_flat = targets.view(-1)  # Shape: (bsz * sl)
                loss = self.loss_fn(logits_flat, targets_flat)
            elif self.task == "associative":
                loss = self.loss_fn(logits, targets)
            else:  # adding task
                loss = self.loss_fn(preds, targets)
            return preds, loss
        else:
            return preds, None

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                # Scale by 4 to account for the sublayers
                self.std *= (4 * self.configs.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            torch.nn.init.uniform_(module.M_u, -self.m_x, self.m_x)
            torch.nn.init.xavier_normal_(module.M_phi_plus)
            torch.nn.init.xavier_normal_(module.M_phi_minus)

            # Initialize Mʸ₂ = α * I, page 8.
            if self.learnable_m_y and module.k_y > 1:
                with torch.no_grad():
                    module.M_y[:, 1] = self.alpha * torch.eye(module.d_model)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Returns:
            int: The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.hybrid.wpe.weight.numel()
        return n_params
