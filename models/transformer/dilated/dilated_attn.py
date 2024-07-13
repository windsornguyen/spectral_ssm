# =============================================================================#
# Authors: Windsor Nguyen
# File: dilated_attn.py
# =============================================================================#

"""
Dilated Attention.
Adapted from torchscale/component/utils.py @ https://github.com/microsoft/torchscale.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.transformer.dilated.multihead_attn import MultiheadAttention
from utils.dist_utils import (
    padding_to_multiple_of, 
    all_gather_func, 
    get_data_parallel_rank, 
    get_data_parallel_world_size
)


class DilatedCausalSelfAttention(nn.Module):
    def __init__(self, configs):
        super(DilatedCausalSelfAttention, self).__init__()
        self.configs = configs
        assert configs.n_embd % configs.n_heads == 0

        # Key, query, value projections for all heads, concatenated
        self.c_attn = nn.Linear(configs.n_embd, 3 * configs.n_embd, bias=configs.bias)

        # The output projection, concatenated
        self.c_proj = nn.Linear(configs.n_embd, configs.n_embd, bias=configs.bias)
        self.c_proj.SCALE_INIT = 1

        # Regularization
        self.dropout = configs.dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.n_embd = configs.n_embd
        self.n_heads = configs.n_heads

        # Flash attention makes the GPUs go brrr, but support is only in PyTorch >= 2.0
        has_flash_attn = hasattr(nn.functional, "scaled_dot_product_attention")
        self.flash_attn = has_flash_attn and configs.flash_attn

        if not self.flash_attn:
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

    def dense_to_sparse(self, x, ratio):
        print(f"Input tensor shape: {x.shape}")
        print(f"Ratio: {ratio}")
        
        length = x.size(1)
        print(f"Length: {length}")
        
        padding = padding_to_multiple_of(length, ratio)
        head_padding = padding_to_multiple_of(self.n_heads, ratio)
        print(f"Padding: {padding}, Head padding: {head_padding}")

        if padding > 0 or head_padding > 0:
            x = F.pad(x, (0, 0, 0, head_padding, 0, padding), value=0.)
            print(f"Shape after padding: {x.shape}")

        print(f"Attempting rearrange with shape: {x.shape}")
        try:
            x = rearrange(x, 'b (l r1) (r2 h) d -> b l h d r1 r2', r1=ratio, r2=ratio)
            print(f"Shape after first rearrange: {x.shape}")
        except Exception as e:
            print(f"Error during first rearrange: {e}")
            print(f"Current tensor shape: {x.shape}")
            print(f"n_heads: {self.n_heads}")
            raise

        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        print(f"Shape after diagonal: {x.shape}")

        x = rearrange(x, 'b l h d r -> b l (r h) d')
        print(f"Shape after second rearrange: {x.shape}")

        if head_padding > 0:
            x = x[:, :, :self.n_heads]
            print(f"Final shape after head padding removal: {x.shape}")

        return x

    def sparse_to_dense(self, out, lse, ratio):
        head_padding = padding_to_multiple_of(self.n_heads, ratio)

        if head_padding > 0:
            out = F.pad(out, (0, 0, 0, head_padding), value = 0.)
            lse = F.pad(lse, (0, 0, 0, head_padding), value = -1e8)

        out = rearrange(out, 'b l (r h) d -> b l h d r', r=ratio)
        out = torch.diag_embed(out, offset=0, dim1=4, dim2=5)
        out = rearrange(out, 'b l h d r1 r2 -> b (r2 h) (l r1) d', r1=ratio, r2=ratio)

        lse = rearrange(lse, 'b (r h) l -> b l h r', r=ratio)
        lse = torch.diag_embed(lse, offset=0, dim1=3, dim2=4)
        lse = lse.masked_fill_(lse==0, -1e8)
        lse = rearrange(lse, 'b l h r1 r2 -> b (r2 h) (l r1) 1', r1=ratio, r2=ratio)

        if head_padding > 0:
            out = out[:, :self.n_heads]
            lse = lse[:, :self.n_heads]

        return out, lse

    def gather_kv(self, x, sl, seq_len, is_causal=True):
        bsz = x.size(0)
        assert sl % seq_len == 0
        num_rank_per_segment = sl // seq_len

        x = all_gather_func(x)
        current_rank = get_data_parallel_rank()
        x = rearrange(x, '(w b) l h d -> w b l h d', b=bsz)
        
        if is_causal:
            if current_rank > 0:
                x = x[:current_rank]
            else:
                x = x[:1] * 0
        
        current_segment = current_rank // num_rank_per_segment * num_rank_per_segment
        x = x[current_segment:current_segment+num_rank_per_segment]

        x = rearrange(x, 'w b l h d -> b (w l) h d')
        return x
    
    def gathering(self, x, dr, sl, is_causal=True, offset=0, is_kv=False, seq_parall=True):
        curr_x = x
        if offset > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, offset % sl, 0), value=0.)
        seq_len = curr_x.size(1)
        should_gather_kv = is_kv and (get_data_parallel_world_size() > 1) and (sl > seq_len) and seq_parall
        _sl = sl
        sl = min(sl, seq_len)
        padding = padding_to_multiple_of(seq_len, sl)

        if padding > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, 0, padding), value = 0.)

        curr_x = rearrange(curr_x, 'b (n g) h d -> (b n) g h d', g=sl)
        curr_x = self.dense_to_sparse(curr_x, dr)

        if should_gather_kv:
            curr_x = self.gather_kv(curr_x, _sl, seq_len, is_causal)

        curr_x = rearrange(curr_x, 'b l h d -> (b h) l d')

        return curr_x

    def scattering(self, outs, lses, seq_len, bsz, offset=0):
        assert len(outs) == len(lses)
        assert len(outs) % len(self.configs.dilated_ratios) == 0
        all_outs, all_lses = [], []
        drs = self.configs.dilated_ratios
        if len(outs) > len(drs):
            drs = drs * (len(outs) // len(drs))

        for dr, o, lse in zip(drs, outs, lses):
            o = rearrange(o, 'b l (h d) -> b l h d', h=self.n_heads)
            o, lse = self.sparse_to_dense(o, lse, dr)
            o = rearrange(o, '(b n) h g d -> (b h) (n g) d', b=bsz)
            lse = rearrange(lse, '(b n) h g 1 -> (b h) (n g) 1', b=bsz)
            o = o[:, offset:offset+seq_len]
            lse = lse[:, offset:offset+seq_len]

            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0)
            max_lse = max_lse.max(0)[0]
            all_lses = [torch.exp(lse-max_lse) for lse in all_lses]
            lse_sum = torch.stack(all_lses, dim=0).sum(0)
            all_lses = [lse / lse_sum for lse in all_lses]

        out = 0
        for o, lse in zip(all_outs, all_lses):
            out += o * lse.type_as(o)
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.n_heads)

        return out

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
        bsz, sl, n_embd = x.size()

        # Combined Q, K, V projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(bsz, sl, self.n_heads, n_embd // self.n_heads).transpose(1, 2)
        q = q.view(bsz, sl, self.n_heads, n_embd // self.n_heads).transpose(1, 2)
        v = v.view(bsz, sl, self.n_heads, n_embd // self.n_heads).transpose(1, 2)

        # Handle incremental state for inference
        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(bsz * self.n_heads, -1, self.n_embd // self.n_heads)
                prev_value = incremental_state["prev_value"].view(bsz * self.n_heads, -1, self.n_embd // self.n_heads)
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(bsz, self.n_heads, -1, self.n_embd // self.n_heads)
            incremental_state["prev_value"] = v.view(bsz, self.n_heads, -1, self.n_embd // self.n_heads)

        outs, lses = [], []
        for sl, dr in zip(self.configs.segment_lengths, self.configs.dilated_ratios):
            ki = self.gathering(k, dr, sl, is_causal=is_causal, offset=0, is_kv=True, seq_parall=self.configs.seq_parallel)
            vi = self.gathering(v, dr, sl, is_causal=is_causal, offset=0, is_kv=True, seq_parall=self.configs.seq_parallel)
            qi = self.gathering(q, dr, sl, is_causal=is_causal, offset=0 if incremental_state is None else sl - 1, is_kv=False, seq_parall=self.configs.seq_parallel)

            if self.flash_attn:
                yi = nn.functional.scaled_dot_product_attention(
                    qi, ki, vi,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=is_causal
                )
                lse = None  # Flash attention doesn't return LSE
            else:
                qi = qi * (ki.size(-1) ** -0.5)
                att = qi @ ki.transpose(-2, -1)
                if is_causal:
                    att = att.masked_fill(self.mask[:, :, :sl, :sl] == 0, float("-inf"))
                att = nn.functional.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                yi = att @ vi
                lse = att.logsumexp(dim=-1, keepdim=True)

            outs.append(yi)
            lses.append(lse)

        attn = self.scattering(outs, lses, sl, bsz, offset=0 if incremental_state is None else sl - 1)

        # Output projection
        attn = self.resid_dropout(self.c_proj(attn))

        return attn, None
