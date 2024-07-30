# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: attn.py
# =============================================================================#

"""Vanilla Self-Attention."""

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Self-attention layer for the Transformer.

    Note: scaled_dot_product_attention enables FlashAttention-2
    (Tri Dao, 2023, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")
    and Memory-Efficient Attention (Rabe et al., 2022, "Self-attention Does Not Need O(n^2) Memory"),
    all written in C++, per the PyTorch documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """

    def __init__(self, configs):
        super(CausalSelfAttention, self).__init__()
        self.configs = configs
        assert configs.d_model % configs.n_heads == 0

        # Key, query, value projections for all heads, concatenated
        self.c_attn = nn.Linear(configs.d_model, 3 * configs.d_model, bias=configs.bias)

        # The output projection, concatenated
        self.c_proj = nn.Linear(configs.d_model, configs.d_model, bias=configs.bias)
        self.c_proj.SCALE_INIT = 1

        # Regularization
        self.dropout = configs.dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads

        # Flash attention makes the GPUs go brrr, but support is only in PyTorch >= 2.0
        has_flash_attn = hasattr(nn.functional, "scaled_dot_product_attention")
        use_flash_attn = has_flash_attn and configs.flash_attn
        self.flash_attn = use_flash_attn

        if not use_flash_attn:
            # TODO: This prints n_layers times. Move it somewhere better.
            if not has_flash_attn:
                print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            else:
                print("WARNING: Using slow attention. Flash Attention is disabled.")
            
            # Manual implementation of the causal mask
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(configs.sl, configs.sl)).view(
                    1, 1, configs.sl, configs.sl
                ),
            )
            

    def forward(self, x):
        """
        Performs the forward pass of the causal self attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (bsz, sl, d_model), where bsz is the batch size,
                sl is the sequence length, and d_model is the embedding dimensionality (d_model).

        Returns:
            torch.Tensor: Output tensor of shape (bsz, sl, d_model) after applying self-attention.
        """
        bsz, sl, d_model = x.size()

        # Compute query, key, values for all heads in batch, and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention
        k = k.view(bsz, sl, self.n_heads, d_model // self.n_heads).transpose(
            1, 2
        )  # -> (B, nh, sl, hs)
        q = q.view(bsz, sl, self.n_heads, d_model // self.n_heads).transpose(
            1, 2
        )  # (B, nh, sl, hs)
        v = v.view(bsz, sl, self.n_heads, d_model // self.n_heads).transpose(
            1, 2
        )  # (B, nh, sl, hs)

        # Causal self-attention; self-attend: (bsz, nh, sl, hs) x (bsz, nh, hs, sl) -> (B, nh, sl, sl)
        if self.flash_attn:
            # Efficient attention using Flash Attention CUDA kernels
            y = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual implementation of self-attention
            q = q * k.size(-1) ** -0.5
            att = q @ k.transpose(-2, -1)
            att = att.masked_fill(self.mask[:, :, :sl, :sl] == 0, float("-inf"))
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (bsz, nh, sl, sl) x (bsz, nh, sl, hs) -> (bsz, nh, sl, hs)

        # Re-assemble / "concat" all attention head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(bsz, sl, d_model)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
