import math

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch import nn
from einops import rearrange
from models.transformer.dilated.multiway_network import MultiwayWrapper
from models.transformer.dilated.xpos_relative_position import XPOS
from models.transformer.dilated.flash_attn import flash_attn_func

try:
    from models.mamba.ops.triton.layer_norm import RMSNorm
except ModuleNotFoundError or ImportError:
    from utils.rms_norm import RMSNorm

 
class MultiheadAttention(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.n_embd = configs.n_embd
        self.n_heads = configs.n_heads
        assert self.n_embd % self.n_heads == 0
        self.head_dim = self.n_embd // self.n_heads
        self.dropout = configs.dropout
        self.bias = configs.bias
        self.scaling = self.head_dim**-0.5

        # Combined projection for Q, K, V
        self.c_attn = MultiwayWrapper(configs, nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.bias))
        
        # Output projection
        self.c_proj = MultiwayWrapper(configs, nn.Linear(self.n_embd, self.n_embd, bias=self.bias))
        
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Optional RMSNorm
        self.inner_attn_rn = MultiwayWrapper(configs, RMSNorm(self.n_embd, eps=configs.rms_norm_eps)) if configs.sub_rn else None

        # Optional XPOS
        self.xpos = XPOS(self.head_dim, configs.xpos_scale_base) if configs.xpos_rel_pos else None

        # Flash attention support
        self.flash_attn = configs.flash_attn and hasattr(nn.functional, "scaled_dot_product_attention")

        if not self.flash_attn:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0 and configs.flash_attn=True")
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(configs.sl, configs.sl)).view(1, 1, configs.sl, configs.sl),
            )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
        if not self.flash_attn:
            q *= self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = torch.nan_to_num(attn_weights)
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                attn_weights = rearrange(attn_weights, '(b h) t s -> b h t s', h=self.n_heads)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = rearrange(attn_weights, 'b h t s -> (b h) t s')

            if rel_pos is not None:
                rel_pos = rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
            attn_probs = self.attn_dropout(attn_weights)

            attn = torch.bmm(attn_probs, v)
            attn = rearrange(attn, '(b h) l d -> b l (h d)', h=self.n_heads)
        else:
            assert flash_attn_func is not None
            assert rel_pos is None
            q = rearrange(q, '(b h) l d -> b l h d', h=self.n_heads)
            k = rearrange(k, '(b h) l d -> b l h d', h=self.n_heads)
            v = rearrange(v, '(b h) l d -> b l h d', h=self.n_heads)
            attn, lse = flash_attn_func(q, k, v, self.dropout, attn_mask, None, is_causal)
            attn = rearrange(attn, 'b l h d -> b l (h d)')
            attn_weights = lse[:, :, :attn.size(1)]

        return attn, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        bsz, sl, _ = x.size()

        # Combined Q, K, V projection
        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        # Reshape for multi-head attention
        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.n_heads)

        # Apply XPOS if configured
        if self.xpos is not None:
            offset = sl - 1 if incremental_state is not None and not is_first_step else 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        # Attention calculation
        y, attn_weights = self.attention_ops(q, k, v, key_padding_mask, attn_mask, is_causal=attn_mask is None)

        # Reshape and apply output projection
        y = rearrange(y, 'b l (h d) -> b l h d', h=self.n_heads)
        y = rearrange(y, 'b l h d -> b l (h d)')
        
        if self.inner_attn_rn is not None:
            y = self.inner_attn_rn(y)

        y = self.resid_dropout(self.c_proj(y))

        return y, attn_weights
