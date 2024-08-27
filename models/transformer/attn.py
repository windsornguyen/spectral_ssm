# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: attn.py
# =============================================================================#

"""Vanilla Self-Attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from flash_attn import flash_attn_func as fa2
except ImportError as e:
    print(f"Unable to import Triton-based flash attention: {e}. No alternative currently available.")


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))
    )

class CausalSelfAttention(nn.Module):
    def __init__(self, configs):
        super(CausalSelfAttention, self).__init__()
        self.configs = configs
        assert (configs.d_model * configs.embd_scale) % configs.n_heads == 0

        self.n_heads = configs.n_heads
        self.d_model = configs.d_model
        self.embd_scale = configs.embd_scale

        # Key, query, value projections for all heads, concatenated
        self.c_attn = nn.Linear(configs.d_model * configs.embd_scale, 3 * configs.d_model * configs.embd_scale, bias=configs.bias)

        # The output projection
        self.c_proj = nn.Linear(configs.d_model * configs.embd_scale, configs.d_model * configs.embd_scale, bias=configs.bias)
        self.c_proj.SCALE_INIT = 1

        # Regularization
        self.dropout = configs.dropout
        self.resid_dropout = nn.Dropout(self.dropout)

        # Flash attention specific
        self.window_size = getattr(configs, 'window_size', 0) # Default to 0 if not specified 
        self.use_alibi = configs.use_alibi
        self.alibi_slopes = self._get_alibi_slopes(self.n_heads)

    def _generate_slopes(self, n: int):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start * (start ** i) for i in range(n)]

    def _get_alibi_slopes(self, n_heads: int, interpolation_factor: float = 0.25):
        if math.log2(n_heads).is_integer():
            slopes = self._generate_slopes(n_heads)
        else:
            n = nearest_power_of_two(n_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][:n_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=self.configs.device)
        slopes = slopes * interpolation_factor
        return slopes

    def forward(self, x):
        bsz, sl, _ = x.size()

        # Compute query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model * self.embd_scale, dim=2)

        # Reshape for multi-head attention
        q = q.view(bsz, sl, self.n_heads, (self.d_model * self.embd_scale) // self.n_heads)
        k = k.view(bsz, sl, self.n_heads, (self.d_model * self.embd_scale) // self.n_heads)
        v = v.view(bsz, sl, self.n_heads, (self.d_model * self.embd_scale) // self.n_heads)

        # Use Flash Attention
        y = fa2(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),
            alibi_slopes=self.alibi_slopes if self.use_alibi else None,
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(bsz, sl, self.d_model * self.embd_scale)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    