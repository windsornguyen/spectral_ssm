# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: mamba.py
# ==============================================================================#

"""The Mamba-2 architecture."""

import math
from typing import Optional

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from utils.nearest_power_of_2 import nearest_power_of_2
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU
from tqdm import tqdm

from models.mamba.mamba import MambaLayer


@dataclass
class Mamba2Configs:
    bsz: int = 2
    n_layers: int = 2
    d_in: int = 10
    d_model: int = 8
    d_out: int = 10
    mlp_scale: int = 4
    embd_scale: int = 1
    dropout: float = 0.10
    d_state: int = 128
    d_conv: int = 4
    conv_init: Optional[float] = None
    expand: int = 2 # The paper sets the expand factor E = 2
    headdim: int = 1
    d_ssm: Optional[int] = None # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
    ngroups: int = 1
    A_init_range: tuple[int, int] = (1, 16)
    activation: str = "silu"
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple[float, float] = (0.0, float("inf"))
    bias: bool = False
    conv_bias: bool = True

    # Fused kernel and sharding options
    chunk_size: int = 256
    use_mem_eff_path: bool = True
    process_group: Optional[any] = None
    seq_parallel: bool = True
    device: Optional[any] = None
    dtype: Optional[any] = None
    world_size: int = 1

    # MoE
    moe: bool = True
    num_experts: int = 8
    num_experts_per_timestep: int = 2

    # TODO: Experiment-specific hyperparameters
    loss_fn: Optional[any] = nn.SiLU()
    task: str = "copy"
    vocab_size: int = 20


class MLP(nn.Module):
    """
    Multi-layer perceptron network using SwiGLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            mlp_scale (float): Scaling factor for hidden dimension.
            d_model (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
    """

    def __init__(self, configs) -> None:
        super(MLP, self).__init__()
        self.h_dim = int(configs.mlp_scale * configs.d_model)
        self.swiglu = SwiGLU(dim=(configs.d_model * configs.embd_scale), h_dim=self.h_dim, bias=configs.bias, use_sq_relu=configs.use_sq_relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.swiglu(x)
        return x

class GatedMLP(nn.Module):
    """
    Gated multi-layer perceptron network using SiLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            d_model (int): Input and output embedding dimension.
            scale (float): Scaling factor for hidden dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super().__init__()
        self.in_features = configs.d_model * configs.embd_scale
        self.out_features = configs.d_model * configs.embd_scale
        self.chunks = 2
        self.hidden_features = int(configs.mlp_scale * configs.d_model)

        self.fc1 = nn.Linear(self.in_features, self.chunks * self.hidden_features, bias=configs.bias)
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=configs.bias)
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
        y = self.fc1(x)
        y, gate = y.chunk(self.chunks, dim=-1)
        y = y * self.silu(gate)
        y = self.fc2(y)
        return self.dropout(y)

class MambaBlock(nn.Module):
    """
    A single block of the spectral SSM model composed of Mamba-2 and MLP layers.

    Args:
        configs: Configuration object for Mamba-2 and MLP layers
        sigma (torch.Tensor): Eigenvalues of the Hankel matrix.
        V (torch.Tensor): Precomputed FFT of top K eigenvectors.
        padded_sl (int): Padded sequence length for FFT operations.
    """

    def __init__(self, configs) -> None:
        super(MambaBlock, self).__init__()
        self.mamba = MambaLayer(configs)
        self.rn = RMSNorm(configs.d_model * configs.embd_scale)
        self.mlp = MoE(
            configs,
            experts=[GatedMLP(configs) for _ in range(configs.num_experts)],
            gate=nn.Linear(configs.d_model * configs.embd_scale, configs.num_experts, bias=configs.bias)
        ) if configs.moe else GatedMLP(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        _, sl, _ = x.size()
        placeholder = self.mamba(x, sl)
        x = x + placeholder
        x = x + self.mlp(self.rn(x))
        return x

class Mamba2(nn.Module):
    """
    Model architecture based on stacked blocks of Mamba-2 and MLP layers.

    Args:
        configs: Configuration object containing model hyperparameters.
    """

    def __init__(self, configs: Mamba2Configs):
        super(Mamba2, self).__init__()
        self.configs = configs
        self.n_layers = self.configs.n_layers
        self.d_in = self.configs.d_in
        self.d_model = self.configs.d_model
        self.d_out = self.configs.d_out
        self.embd_scale = self.configs.embd_scale
        self.bias = self.configs.bias
        self.loss_fn = self.configs.loss_fn
        self.task = self.configs.task

        if configs.moe:
            print(f"\nMoE?: Enabled | Using {configs.num_experts} experts.")
        else:
            print("\nMoE?: Disabled")

        if configs.task == "adding":
            self.embed = nn.Linear(configs.d_in, self.d_model * self.embd_scale)
        elif configs.task in ["copy", "induction"]:
            self.embed = nn.Embedding(configs.d_in, self.d_model * self.embd_scale)

        self.mamba = nn.ModuleDict(
            dict(
                hidden=nn.ModuleList(
                    [MambaBlock(self.configs) for _ in range(self.n_layers)]),
            )
        )
        self.output = nn.Linear(self.d_model * self.embd_scale, self.d_out, bias=self.bias)

        # Report the number of parameters
        print(
            "Mamba-2 Model Parameter Count (excl. pos. emb.): %.2fM"
            % (self.get_num_params() / 1e6,)
        )

    def get_num_params(self):
        """
        Return the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def forward(self, inputs, targets):
        """
        Forward pass of the Mamba-2 model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor): Target tensor for loss computation

        Returns:
            Type (ignore due to high variability):
            - Predictions tensor
            - tuple containing loss and metrics (if applicable)
        """
        # Embed the input categories
        if self.task in ["copy", "induction", "associative"]:
            x = self.embed(inputs)  # Shape: (bsz, sl, d_model)
        elif self.task == "adding":
            # Reshape inputs from (bsz, sl * 2) to (bsz, sl, 2)
            x = inputs.view(inputs.shape[0], -1, self.configs.d_in)
            x = self.embed(x)  # Shape: (bsz, sl, d_model)

        # Apply the spectral SSM layers
        for layer in self.mamba.hidden:
            x = layer(x)

        # For associative recall, we only need to predict based on the last token
        if self.task == "associative":
            x = x[:, -1, :]  # Shape: (bsz, d_model)
            logits = self.output(x)  # Shape: (bsz, d_out)
        else:
            logits = self.output(x)  # Shape: (bsz, sl, d_out)

        # Compute predictions
        if self.task in ["copy", "induction", "associative"]:
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
