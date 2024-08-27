# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: transformer.py
# =============================================================================#

"""The Transformer architecture for benchmark tasks."""

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from spectral_ssm.models.transformer.attn_old import CausalSelfAttention
from models.transformer.dilated.dilated_attn import DilatedCausalSelfAttention
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU


@dataclass
class TransformerConfigs:
    n_layers: int = 2
    d_model: int = 8  # Embedding dimension
    n_heads: int = 16  # Constraint: n_heads % d_model == 0
    d_in: int = 10
    d_out: int = 10
    sl: int = 1000  # Sequence length
    mlp_scale: int = 8
    embd_scale: int = 1
    sub_rn: bool = True
    bias: bool = False
    dropout: float = 0.10
    flash_attn: bool = True
    use_sq_relu: bool = False
    task: str = "copy"
    vocab_size: int = 20
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    device: torch.device = None

    # MoE
    moe: bool = True
    num_experts: int = 8
    num_experts_per_timestep: int = 2

    # Dilated Attention
    dilated_attn: bool = False
    segment_lengths: list[int] = field(
        default_factory=lambda: [128]
    )  # TODO: Check this makes sense (and follows paper)
    dilated_ratios: list[int] = field(
        default_factory=lambda: [1]
    )  # TODO: Check this makes sense (and follows paper)
    seq_parallel: bool = True
    xpos_rel_pos: bool = True
    xpos_scale_base: int = 512
    rms_norm_eps: float = 1e-5
    multiway: bool = False


class FFN(nn.Module):
    """
    Feed-forward network using the SwiGLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            mlp_scale (float): Scaling factor for hidden dimension.
            d_model (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super(FFN, self).__init__()
        self.swiglu = SwiGLU(
            dim=configs.d_model * configs.embd_scale,
            h_dim=configs.mlp_scale * configs.d_model,
            bias=configs.bias,
            use_sq_relu=configs.use_sq_relu,
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
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


class GatedFFN(nn.Module):
    """
    Gated feed-forward network using SiLU activation.

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
        Forward pass of the GatedFFN.

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


class TransformerBlock(nn.Module):
    """
    Single block of the Transformer.
    """

    def __init__(self, configs):
        super(TransformerBlock, self).__init__()
        self.configs = configs
        self.rn_1 = RMSNorm(configs.d_model * configs.embd_scale, eps=configs.rms_norm_eps)
        self.rn_2 = RMSNorm(configs.d_model * configs.embd_scale, eps=configs.rms_norm_eps)
        self.attn = self._get_attn_type(configs)

        self.ffn = (
            MoE(
                configs,
                experts=[GatedFFN(configs) for _ in range(configs.num_experts)],
                gate=nn.Linear(configs.d_model * configs.embd_scale, configs.num_experts, bias=configs.bias),
            )
            if configs.moe
            else GatedFFN(configs)
        )

    def _get_attn_type(self, configs):
        if configs.dilated_attn:
            return DilatedCausalSelfAttention(configs)
        else:
            return CausalSelfAttention(configs)

    def forward(self, x):
        x = x + self.attn(self.rn_1(x))
        x = x + self.ffn(self.rn_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer architecture adapted from the GPT-2 implementation.
    """

    def __init__(self, configs):
        super(Transformer, self).__init__()
        assert configs.sl is not None
        self.configs = configs
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(self.configs.dropout)

        if configs.moe:
            print(f"\nMoE?: Enabled | Using {configs.num_experts} experts.")
        else:
            print("\nMoE?: Disabled")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(configs.d_in, configs.d_model * configs.embd_scale),
                wpe=nn.Embedding(configs.sl, configs.d_model * configs.embd_scale),
                dropout=self.dropout,
                hidden=nn.ModuleList(
                    [TransformerBlock(configs) for _ in range(configs.n_layers)]
                ),
            )
        )

        # Add an embedding layer for the copying task
        if configs.task == "adding":
            self.embed = nn.Linear(configs.d_in, self.d_model * configs.embd_scale)
        elif configs.task in ["copy", "induction", "associative"]:
            self.embed = nn.Embedding(configs.d_in, self.d_model * configs.embd_scale)

        self.output = nn.Linear(configs.d_model * configs.embd_scale, configs.d_out, bias=configs.bias)
        self.loss_fn = self.configs.loss_fn

        # Initialize all weights
        self.std = (self.d_model * configs.embd_scale)**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print(
            "\nTransformer Model Parameter Count (excl. pos. emb.): %.2fM"
            % (self.get_num_params() / 1e6,)
        )

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                # Scale by 2 to account for self-attn and ffn sub-layer
                self.std *= (2 * self.configs.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)

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
            n_params -= (
                self.transformer.wte.weight.numel()
                + self.transformer.wpe.weight.numel()
            )
        return n_params

    def forward(self, inputs, targets=None):
        """
        Perform the forward pass of the Transformer model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor, optional): Target tensor for training

        Returns:
            torch.Tensor: Predicted output tensor of shape (bsz, sl, d_out)
            tuple: Loss (and metrics, if applicable)
        """
        bsz, sl = inputs.size()
        assert (
            sl <= self.configs.sl
        ), f"Input sequence length {sl} exceeds model's maximum sequence length {self.configs.sl}"

        # Embed the input categories
        if self.configs.task in ["copy", "induction", "associative"]:
            x = self.embed(inputs)  # Shape: (bsz, sl, d_model)
        elif self.configs.task == "adding":
            # Reshape inputs from (bsz, sl * 2) to (bsz, sl, 2)
            x = inputs.view(inputs.shape[0], -1, self.configs.d_in)
            x = self.embed(x)  # Shape: (bsz, sl, d_model)

        # Position embeddings
        pos = torch.arange(0, sl, dtype=torch.long, device=inputs.device).unsqueeze(0)
        pos_embeddings = self.transformer.wpe(pos)

        # Combine token and position embeddings
        x = x + pos_embeddings
        x = self.transformer.dropout(x)

        # Pass through transformer layers
        for layer in self.transformer.hidden:
            x = layer(x)

        # Generate logits
        if self.configs.task == "associative":
            x = x[:, -1, :]  # Shape: (bsz, d_model)
            logits = self.output(x)  # Shape: (bsz, d_out)
        else:
            logits = self.output(x)  # Shape: (bsz, sl, d_out)

        # Compute predictions
        if self.configs.task in ["copy", "induction", "associative"]:
            preds = torch.argmax(logits, dim=-1)
        elif self.configs.task == "adding":
            preds = logits.mean(dim=1).squeeze(-1)

        if targets is not None:
            if self.configs.task == "copy":
                loss = self.loss_fn(logits.view(-1, self.d_out), targets.view(-1))
            elif self.configs.task == "induction":
                logits_flat = logits.view(
                    -1, logits.size(-1)
                )  # Shape: (bsz * sl, vocab_size)
                targets_flat = targets.view(-1)  # Shape: (bsz * sl)
                loss = self.loss_fn(logits_flat, targets_flat)
            elif self.configs.task == "associative":
                loss = self.loss_fn(logits, targets)
            else:  # adding task
                loss = self.loss_fn(preds, targets)
            return preds, loss
        else:
            return preds, None
