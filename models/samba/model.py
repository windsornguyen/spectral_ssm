# ==============================================================================#
# Authors: Windsor Nguyen
# File: model.py (Samba)
# ==============================================================================#

"""
The Samba model, a hybrid state-space-attention model as introduced in
"Samba: Simple Hybrid State Space Models for Efficient Unlimited 
Context Language Modeling" by Ren et al. (2024).
"""

import math
from typing import Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import rearrange

from models.transformer.model import CausalSelfAttention
from mamba.models.mixer_seq_simple import Mamba, MambaConfig
from mamba.ops.triton.layernorm import RMSNorm
from utils.swiglu import SwiGLU


@dataclass
class JambaConfigs:
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    n_layers: int = 6
    n_heads: int = 8
    d_head: int = 64
    expand: int = 2
    d_inner: Optional[int] = None
    use_moe: bool = False
    num_experts: int = 4
    top_k: int = 2
    conv_bias: bool = True
    bias: bool = False
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation: str = "silu"
    vocab_size: int = 50000
    mamba_config: Optional[MambaConfig] = None
    device: Any = None
    dtype: Any = None
    loss_fn: nn.Module = field(default_factory=lambda: nn.CrossEntropyLoss())
    controls: dict = field(
        default_factory=lambda: {"task": "language-modeling", "model": "jamba"}
    )


class JambaLayer(nn.Module):
    """
    A single Jamba layer combining Mamba and MLP/MoE.

    Args:
        configs: Configuration object for the Jamba layer
    """

    def __init__(self, configs: JambaConfigs) -> None:
        super(JambaLayer, self).__init__()
        self.configs = configs
        self.use_moe = configs.use_moe

        # Mamba layer
        if configs.mamba_config is None:
            configs.mamba_config = MambaConfig(
                d_model=configs.d_model,
                d_state=configs.d_state,
                d_conv=configs.d_conv,
                expand=configs.expand,
            )
        self.mamba = Mamba(configs.mamba_config)

        # MLP or MoE layer
        if self.use_moe:
            self.moe = MoE(configs)
        else:
            self.mlp = MLP(configs)

        # Normalization layers
        self.norm1 = RMSNorm(configs.d_model, eps=configs.layer_norm_epsilon)
        self.norm2 = RMSNorm(configs.d_model, eps=configs.layer_norm_epsilon)

        # Dropout
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Jamba layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Mamba branch
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual

        # MLP or MoE branch
        residual = x
        x = self.norm2(x)
        if self.use_moe:
            x = self.moe(x)
        else:
            x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual

        return x


class MambaAttentionMoELayer(nn.Module):
    """
    A single layer combining Mamba, Attention, and MLP/MoE.

    Args:
        configs: Configuration object for the layer
    """

    def __init__(self, configs: JambaConfigs) -> None:
        super(MambaAttentionMoELayer, self).__init__()
        self.configs = configs
        self.use_moe = configs.use_moe

        # Mamba layer
        if configs.mamba_config is None:
            configs.mamba_config = MambaConfig(
                d_model=configs.d_model,
                d_state=configs.d_state,
                d_conv=configs.d_conv,
                expand=configs.expand,
            )
        self.mamba = Mamba(configs.mamba_config)

        # Attention layer
        self.attention = CausalSelfAttention(configs)

        # MLP or MoE layer
        if self.use_moe:
            self.moe = MoE(configs)
        else:
            self.mlp = MLP(configs)

        # Normalization layers
        self.norm1 = RMSNorm(configs.d_model, eps=configs.layer_norm_epsilon)
        self.norm2 = RMSNorm(configs.d_model, eps=configs.layer_norm_epsilon)
        self.norm3 = RMSNorm(configs.d_model, eps=configs.layer_norm_epsilon)

        # Dropout
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mamba-Attention-MoE layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Mamba branch
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual

        # Attention branch
        residual = x
        x = self.norm2(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual

        # MLP or MoE branch
        residual = x
        x = self.norm3(x)
        if self.use_moe:
            x = self.moe(x)
        else:
            x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual

        return x


class MLP(nn.Module):
    """
    Simple multi-layer perceptron network using SwiGLU activation.

    Args:
        configs: Configuration object for the MLP
    """

    def __init__(self, configs: JambaConfigs) -> None:
        super(MLP, self).__init__()
        self.d_inner = (
            configs.d_inner if configs.d_inner is not None else 4 * configs.d_model
        )
        self.swiglu = SwiGLU(dim=configs.d_model, h_dim=self.d_inner, bias=configs.bias)
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
        return self.dropout(x)


class MoE(nn.Module):
    """
    Mixture of Experts layer.

    Args:
        configs: Configuration object for the MoE layer
    """

    def __init__(self, configs: JambaConfigs) -> None:
        super(MoE, self).__init__()
        self.configs = configs
        self.num_experts = configs.num_experts
        self.top_k = configs.top_k
        self.d_model = configs.d_model
        self.d_inner = (
            configs.d_inner if configs.d_inner is not None else 4 * configs.d_model
        )

        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(configs) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        original_shape = x.shape
        x = rearrange(x, "b s d -> (b s) d")

        # Gate computation
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        # Expert computation
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_mask = indices.eq(i).any(dim=-1)
            if expert_mask.any():
                expert_inputs = x[expert_mask]
                expert_outputs[expert_mask] += expert(expert_inputs)

        # Combine expert outputs
        output = torch.einsum("bke,bd->bde", weights, expert_outputs)
        output = output.sum(dim=1)

        return output.view(original_shape)


class Jamba(nn.Module):
    """
    Jamba model architecture combining Mamba and Attention layers.

    Args:
        configs: Configuration object containing model parameters
    """

    def __init__(self, configs: JambaConfigs) -> None:
        super(Jamba, self).__init__()
        self.configs = configs
        self.n_layers = configs.n_layers
        self.d_model = configs.d_model
        self.vocab_size = configs.vocab_size
        self.use_moe = configs.use_moe

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList(
            [
                JambaLayer(configs) if i % 2 == 0 else MambaAttentionMoELayer(configs)
                for i in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(self.d_model, eps=configs.layer_norm_epsilon)

        # Initialize all weights
        self.apply(self._init_weights)

        # Report the number of parameters
        print("Jamba Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the Jamba model.

        Args:
            input_ids (torch.Tensor): Input token ids
            targets (torch.Tensor, optional): Target token ids for loss computation

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output logits
                - Loss (if targets are provided)
        """
        x = self.embed(input_ids)
        x /= math.sqrt(self.d_model)  # Scale embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = torch.matmul(x, self.embed.weight.transpose(0, 1))

        loss = None
        if targets is not None:
            loss = self.configs.loss_fn(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )

        return logits, loss

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding (bool, optional): Whether to exclude embedding parameters. Defaults to True.

        Returns:
            int: The number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        return n_params

    # TODO: Implement other utility methods like `estimate_mfu`, `flops_per_token`, etc.
