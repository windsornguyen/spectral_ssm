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
        self.n_embd = self.configs.n_embd
        self.n_heads = self.configs.n_heads
        assert self.n_embd % self.n_heads == 0
        self.head_dim = self.configs.n_embd // self.n_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = self.configs.dropout
        self.sub_rn = self.configs.sub_rn
        self.bias = self.configs.bias

        self.k_proj = MultiwayWrapper(configs, nn.Linear(self.n_embd, self.n_embd, bias=self.bias))
        self.v_proj = MultiwayWrapper(configs, nn.Linear(self.n_embd, self.n_embd, bias=self.bias))
        self.q_proj = MultiwayWrapper(configs, nn.Linear(self.n_embd, self.n_embd, bias=self.bias))
        self.out_proj = MultiwayWrapper(
            self.configs, nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        )
        self.inner_attn_rn = (
            MultiwayWrapper(self.configs, RMSNorm(self.n_embd, eps=self.configs.rms_norm_eps))
            if self.sub_rn
            else None
        )
        self.dropout_module = torch.nn.Dropout(self.dropout)
        self.xpos = (
            XPOS(self.head_dim, self.configs.xpos_scale_base)
            if self.configs.xpos_rel_pos
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
        if not self.configs.flash_attn:
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

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                attn_weights
            )
            attn_probs = self.dropout_module(attn_weights)

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
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        bsz, tgt_len, n_embd = query.size()
        src_len = tgt_len
        assert n_embd == self.n_embd, f"query dim {n_embd} != {self.n_embd}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.n_heads)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.n_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.n_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.n_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.n_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn, attn_weights = self.attention_ops(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rel_pos=rel_pos, is_causal=is_causal)

        if self.inner_attn_rn is not None:
            attn = self.inner_attn_rn(attn)

        attn = self.out_proj(attn)

        return attn, attn_weights
