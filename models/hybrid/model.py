# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: model.py
# ==============================================================================#

"""The Spectral State Space Model architecture with attention."""

import math

import torch
import torch.nn as nn
from typing import Optional
from flashfftconv import FlashFFTConv

from dataclasses import dataclass, field
from models.stu.stu_utils import (
    get_spectral_filters,
    preconvolve,
    convolve,
    flash_convolve,
)
from models.transformer.attn import CausalSelfAttention
from utils.nearest_power_of_2 import nearest_power_of_2
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU
from tqdm import tqdm
from torch.nn import MSELoss


@dataclass
class SpectralHybridConfigs:
    # STU settings
    d_in: int = 37
    d_out: int = 29
    mlp_scale: int = 4
    embd_scale: int = 1
    num_eigh: int = 16
    k_y: int = 2  # Number of parametrizable, autoregressive matrices Mʸ
    k_u: int = 3  # Number of parametrizable, autoregressive matrices Mᵘ
    alpha: float = 0.9  # 0.9 deemed "uniformly optimal" in the paper
    use_ar_y: bool = False
    use_ar_u: bool = False
    use_hankel_L: bool = False
    use_flash_fft: bool = True
    use_approx: bool = True
    
    # Transformer settings
    d_model: int = 37 # Constraint: n_heads % d_model == 0
    n_heads: int = 16 # Constraint: n_heads % d_model == 0
    flash_attn: bool = True
    use_sq_relu: bool = False
    
    # MoE
    moe: bool = True
    num_experts: int = 8
    num_experts_per_timestep: int = 2

    # Dilated Attention settings
    sub_rn: bool = True
    flash_attn: bool = True
    dilated_attn: bool = False
    segment_lengths: list[int] = field(default_factory=lambda: [128])
    dilated_ratios: list[int] = field(default_factory=lambda: [1])
    seq_parallel: bool = True
    xpos_rel_pos: bool = True
    xpos_scale_base: int = 512
    rms_norm_eps: float = 1e-5
    multiway: bool = False

    # General training settings
    sl: int = 1_000 # Sequence length
    n_layers: int = 2
    bias: bool = False
    dropout: float = 0.10
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )
    device: torch.device = None


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

    def __init__(self, configs: SpectralHybridConfigs, phi: torch.Tensor, n: int, flash_fft: FlashFFTConv = None) -> None:
        super(STU, self).__init__()
        self.configs = configs
        self.phi = phi
        self.n = n
        self.flash_fft = flash_fft
        self.use_flash_fft = configs.use_flash_fft
        self.K = configs.num_eigh
        self.d_in = configs.d_model
        self.d_out = configs.d_model
        self.use_hankel_L = configs.use_hankel_L
        self.use_approx = configs.use_approx
        
        if self.use_approx:
            self.M_inputs = nn.Parameter(torch.empty(self.d_in, self.d_out))
            self.M_filters = nn.Parameter(torch.empty(self.K, self.d_in))
        else:
            self.M_phi_plus = nn.Parameter(torch.empty(self.K, self.d_in, self.d_out))
            self.M_phi_minus = nn.Parameter(torch.empty(self.K, self.d_in, self.d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_in = x.shape

        if self.use_approx:
            # Contract inputs and filters over the K and d_in dimensions, then convolve
            x_proj = x @ self.M_inputs
            phi_proj = self.phi @ self.M_filters
            if self.use_flash_fft and self.flash_fft is not None: 
                spectral_plus, spectral_minus = flash_convolve(x_proj, phi_proj, self.flash_fft, self.use_approx)
            else:
                spectral_plus, spectral_minus = convolve(x_proj, phi_proj, self.n, self.use_approx)
        else:
            # Convolve inputs and filters,
            if self.use_flash_fft and self.flash_fft is not None: 
                U_plus, U_minus = flash_convolve(x, self.phi, self.flash_fft, self.use_approx)
            else:
                U_plus, U_minus = convolve(x, self.phi, self.n, self.use_approx)
            # Then, contract over the K and d_in dimensions
            spectral_plus = torch.tensordot(U_plus, self.M_phi_plus, dims=([2, 3], [0, 1]))
            spectral_minus = torch.tensordot(U_minus, self.M_phi_minus, dims=([2, 3], [0, 1]))

        return spectral_plus + spectral_minus

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
            dim=configs.d_model * configs.embd_scale, h_dim=int(configs.mlp_scale * configs.d_model),
            bias=configs.bias, use_sq_relu=configs.use_sq_relu
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
        self.in_features = configs.d_model * configs.embd_scale
        self.out_features = configs.d_model * configs.embd_scale
        self.chunks = 2
        self.hidden_features = int(configs.mlp_scale * configs.d_model)

        self.fc_1 = nn.Linear(self.in_features, self.chunks * self.hidden_features, bias=configs.bias)
        self.fc_2 = nn.Linear(self.hidden_features, self.out_features, bias=configs.bias)
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

    def __init__(self, configs: SpectralHybridConfigs, phi: torch.Tensor, n: int, flash_fft: FlashFFTConv = None) -> None:
        super(HybridBlock, self).__init__()
        self.stu = STU(configs, phi, n, flash_fft)
        self.attn = CausalSelfAttention(configs)

        self.mlp_1 = MoE(
            configs,
            experts=[GatedMLP(configs) for _ in range(configs.num_experts)],
            gate=nn.Linear(configs.d_model * configs.embd_scale, configs.num_experts, bias=configs.bias)
        ) if configs.moe else GatedMLP(configs)

        self.mlp_2 = MoE(
            configs,
            experts=[GatedMLP(configs) for _ in range(configs.num_experts)],
            gate=nn.Linear(configs.d_model * configs.embd_scale, configs.num_experts, bias=configs.bias)
        ) if configs.moe else GatedMLP(configs)

        self.rn_1 = RMSNorm(configs.d_model * configs.embd_scale)
        self.rn_2 = RMSNorm(configs.d_model * configs.embd_scale)
        self.rn_3 = RMSNorm(configs.d_model * configs.embd_scale)
        self.rn_4 = RMSNorm(configs.d_model * configs.embd_scale)

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

    def __init__(self, configs: SpectralHybridConfigs) -> None:
        super(SpectralHybrid, self).__init__()
        self.configs = configs
        self.n_layers = configs.n_layers
        self.d_model = configs.d_model
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.embd_scale = configs.embd_scale
        self.sl = configs.sl
        self.num_eigh = configs.num_eigh
        self.alpha = configs.alpha
        self.use_hankel_L = configs.use_hankel_L
        self.device = configs.device

        self.phi = get_spectral_filters(self.sl, self.num_eigh, self.use_hankel_L)
        self.n = nearest_power_of_2(self.sl * 2 - 1)
        self.V = preconvolve(self.phi, self.n, configs.use_approx)

        self.bias = configs.bias
        self.dropout = configs.dropout
        self.controls = configs.controls
        
        self.hybrid = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(self.sl, self.d_model * configs.embd_scale),
                dropout=nn.Dropout(configs.dropout),
                hidden=nn.ModuleList(
                    [HybridBlock(
                        self.configs, self.phi, self.n, FlashFFTConv(configs.sl) if configs.use_flash_fft else None
                    ) for _ in range(self.n_layers)]),
            )
        )
        
        self.input_proj = nn.Linear(self.d_in, self.d_model * self.embd_scale, bias=self.bias)
        self.output_proj = nn.Linear(self.d_model * self.embd_scale, self.d_out, bias=configs.bias)
        self.loss_fn = self.configs.loss_fn
        
        # Initialize all weights
        self.m_x = self.d_out**-0.5
        self.std = (self.d_model * self.embd_scale)**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print("\nSTU-Attention Hybrid Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, inputs, targets):
        """
        Forward pass of the spectral SSM model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor): Target tensor for loss computation

        Returns:
            Type (ignore due to high variability):
            - Predictions tensor
            - Tuple containing loss and metrics (if applicable)
        """
        _, sl, _ = inputs.size()

        x = self.input_proj(inputs)
        
        # Generate positional embeddings for the sequence
        pos = torch.arange(0, sl, dtype=torch.long, device=inputs.device)  # -> (sl)

        # Position embeddings of shape (sl, d_model)
        pos_emb = self.hybrid.wpe(pos)  # -> (sl, d_model)

        # Add positional embeddings to the input
        x = x + pos_emb
        
        x = self.hybrid.dropout(x)
        for block in self.hybrid.hidden:
            x = block(x)
        preds = self.output_proj(x)

        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            return preds, (loss,)

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
            if module.use_approx:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                torch.nn.init.xavier_normal_(module.M_phi_minus)

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

    def estimate_mfu(self, fwdbwd_per_iter: float, dt: float) -> tuple[float, float]:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        Args:
            fwdbwd_per_iter (float): Number of forward/backward passes per iteration.
            dt (float): Time taken for the iteration.

        Returns:
            tuple[float, float]: Estimated MFU and estimated peak FLOPS.

        Reference:
            PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        """
        cfg = self.configs
        L, D, E, T = cfg.n_layers, cfg.d_model, cfg.num_eigh, cfg.sl

        total_flops = 0

        # STU blocks
        for _ in range(L):
            total_flops += self._compute_rmsnorm_flops(D, T) * 2  # 2 RMSNorm per block
            total_flops += self._compute_spectral_flops(T, D, E)
            
            if cfg.use_ar_u:
                total_flops += 2 * cfg.k_u * D * D * T  # compute_ar_u
            if cfg.use_ar_y:
                total_flops += self._compute_ar_y_flops(T, D, cfg.k_y)

            total_flops += self._compute_swiglu_flops(D, cfg.scale, T)

        # Output layer
        total_flops += 2 * D * cfg.d_out * T

        # Dropout operations (1 FLOP per element)
        total_flops += (L + 1) * D * T  # L blocks + initial dropout

        flops_per_iter = total_flops
        flops_achieved = flops_per_iter * fwdbwd_per_iter / dt
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops (312 TFLOPS)
        return flops_achieved, flops_promised

    def _compute_rmsnorm_flops(self, D: int, T: int) -> int:
        """
        Compute FLOPS for RMSNorm operation.

        Args:
            D (int): Embedding dimension.
            T (int): Sequence length.

        Returns:
            int: Total FLOPS for RMSNorm.
        """
        power_flops = D * T  # x^2
        sum_flops = D * T  # Sum across dimension
        mean_flops = T  # Division for mean
        sqrt_flops = T  # Square root
        normalize_flops = D * T  # Division by sqrt(mean + eps)
        scale_flops = D * T  # Multiplication by weight

        return power_flops + sum_flops + mean_flops + sqrt_flops + normalize_flops + scale_flops

    def _compute_spectral_flops(self, T: int, D: int, E: int) -> int:
        """
        Compute FLOPS for spectral computations.

        Args:
            T (int): Sequence length.
            D (int): Embedding dimension.
            E (int): Number of eigenvectors.

        Returns:
            int: Total FLOPS for spectral computations.
        """
        n = nearest_power_of_2(T * 2 - 1)
        log_n = math.log2(n)

        fft_flops = 2 * n * log_n * D * E * 2  # rfft and irfft
        filter_flops = 2 * T * D * E  # Spectral filter application

        return int(fft_flops + filter_flops)

    def _compute_ar_y_flops(self, T: int, D: int, k_y: int) -> int:
        """
        Compute FLOPS for compute_ar_y function.

        Args:
            T (int): Sequence length.
            D (int): Embedding dimension.
            k_y (int): Autoregressive depth on the output sequence.

        Returns:
            int: Total FLOPS for ar_y computation.
        """
        return 2 * T * k_y * D * D + T * k_y * D  # Matrix multiplications + additions

    def _compute_swiglu_flops(self, D: int, scale: float, T: int) -> int:
        """
        Compute FLOPS for SwiGLU activation.

        Args:
            D (int): Embedding dimension.
            scale (float): Scaling factor for hidden dimension.
            T (int): Sequence length.

        Returns:
            int: Total FLOPS for SwiGLU computation.
        """
        h_dim = int(scale * D)
        
        linear_flops = 2 * D * h_dim * T * 3  # Three linear layers
        silu_flops = 3 * h_dim * T  # SiLU activation (3 ops per element)
        hadamard_flops = h_dim * T  # Hadamard product
        
        return linear_flops + silu_flops + hadamard_flops

    def predict_states(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 950,
        steps: int = 50,
        rollout_steps: int = 20,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]
        ],
    ]:
        """
        Perform autoregressive prediction with optional periodic grounding to true targets.

        Args:
            inputs (torch.Tensor): Input tensor of shape (num_traj, total_steps, d_in)
            targets (torch.Tensor): Target tensor of shape (num_traj, total_steps, d_out)
            init (int): Index of the initial state to start the prediction
            steps (int): Number of steps to predict

        Returns:
        tuple: Contains the following elements:
            - preds (torch.Tensor): Predictions of shape (num_traj, total_steps, d_out)
            - tuple:
                - avg_loss (torch.Tensor): Scalar tensor with the average loss
                - traj_losses (torch.Tensor): Losses for each trajectory and step, shape (num_traj, steps)
        """
        device = next(self.parameters()).device
        print(f"Predicting on {device}.")
        num_traj, total_steps, d_out = targets.size()
        assert init + steps <= total_steps, f"Cannot take more steps than {total_steps}"
        assert rollout_steps <= steps, f"Cannot roll out for more than total steps"

        # Track model hallucinations
        predicted_steps = torch.zeros(num_traj, steps, d_out, device=device)

        # Track loss between rollout vs ground truth
        traj_losses = torch.zeros(num_traj, steps, device=device)

        # Initialize cost function
        mse_loss = nn.MSELoss()

        for step in tqdm(range(steps), desc="Predicting", unit="step"):
            current_step = init + step

            # Predict the next state using a fixed window size of inputs
            step_preds, (_, _) = self.forward(
                inputs[:, :current_step], targets[:, :current_step]
            )

            # Calculate the mean loss of the last rollout_steps predictions
            rollout_preds = step_preds[:, -rollout_steps:, :]
            rollout_ground_truths = targets[:, (current_step - rollout_preds.shape[1]) : current_step, :]
            traj_losses[:, step] = mse_loss(rollout_preds, rollout_ground_truths)

            # Store the last prediction step for plotting
            predicted_steps[:, step] = step_preds[:, -1].squeeze(1)

            # # Concatenate the autoregressive predictions of states and the ground truth actions (hallucinating)
            # next_action = inputs[:, current_step:current_step+1, -(d_in - d_out):]
            # next_input = torch.cat([next_input, next_action], dim=2)
            # ar_inputs = torch.cat([ar_inputs, next_input], dim=1)

        avg_loss = traj_losses.mean()
        return predicted_steps, (avg_loss, traj_losses)
    