# =============================================================================#
# Authors: Tri Dao, Albert Gu, Windsor Nguyen, Isabel Liu
# File: (Mamba-2) layer.py
# =============================================================================#

"""
Adapted from
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
"""
from typing import Optional
from dataclasses import dataclass, field

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
    
from models.mamba.distributed.distributed_utils import all_reduce, reduce_scatter
from models.mamba.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)
from models.mamba.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from models.mamba.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

@dataclass
class Mamba2Configs:
    bsz: int = 8
    n_layers: int = 2
    d_model: int = 32
    d_out: int = 29
    d_state: int = 128
    d_conv: int = 4
    conv_init: Optional[float] = None
    expand: int = 2 # The paper sets the expand factor E = 2
    headdim: int = 64
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
    sequence_parallel: bool = True
    device: Optional[any] = None
    dtype: Optional[any] = None
    world_size: int = 1

    # TODO: Experiment-specific hyperparameters
    loss_fn: Optional[any] = nn.SiLU()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )

class MambaLayer(nn.Module):
    def __init__(self, configs: Mamba2Configs):
        super(MambaLayer, self).__init__()
        self.configs = configs

        self.bsz = configs.bsz
        self.ngroups = self.configs.ngroups // self.configs.world_size

        # _init_dimensions
        self.d_inner = (self.configs.expand * self.configs.d_model) // self.configs.world_size
        assert (self.d_inner * self.configs.world_size == self.configs.expand * self.configs.d_model)

        self.d_ssm = (
            self.d_inner
            if self.configs.d_ssm is None
            else self.configs.d_ssm // self.configs.world_size
        )
        assert self.d_ssm % self.configs.headdim == 0
        self.nheads = self.d_ssm // self.configs.headdim
        assert self.configs.ngroups % self.configs.world_size == 0

        # _init_layers
        d_in_proj = (2 * self.d_inner + 2 * self.ngroups * self.configs.d_state + self.nheads)

        if self.configs.process_group is None:
            self.in_proj = nn.Linear(
                self.configs.d_model,
                d_in_proj,
                bias=self.configs.bias,
                device=self.configs.device,
                dtype=self.configs.dtype,
            )
        else:
            self.in_proj = ColumnParallelLinear(
                self.configs.d_model,
                d_in_proj * self.configs.world_size,
                bias=self.configs.bias,
                process_group=self.configs.process_group,
                sequence_parallel=self.configs.sequence_parallel,
                device=self.configs.device,
                dtype=self.configs.dtype,
            )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.configs.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=self.configs.conv_bias,
            kernel_size=self.configs.d_conv,
            groups=conv_dim,
            padding=self.configs.d_conv - 1,
            device=self.configs.device,
            dtype=self.configs.dtype,
        )

        if self.configs.rmsnorm:
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.configs.norm_before_gate,
                group_size=self.d_ssm // self.configs.ngroups,
                device=self.configs.device,
                dtype=self.configs.dtype,
            )

        if self.configs.process_group is None:
            self.out_proj = nn.Linear(
                self.d_inner,
                self.configs.d_out,
                bias=self.configs.bias,
                device=self.configs.device,
                dtype=self.configs.dtype,
            )
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.configs.world_size,
                self.configs.d_out,
                bias=self.configs.bias,
                process_group=self.configs.process_group,
                sequence_parallel=self.configs.sequence_parallel,
                device=self.configs.device,
                dtype=self.configs.dtype,
            )

        # _initialize_states
        conv_dtype = self.conv1d.weight.dtype
        ssm_dtype = self.in_proj.weight.dtype
        
        self.conv_state = torch.zeros(
            self.bsz, self.configs.d_conv, self.conv1d.weight.shape[0],
            device=self.conv1d.weight.device, dtype=conv_dtype
        ).transpose(1, 2)
        
        self.ssm_state = torch.zeros(
            self.bsz, self.nheads, self.configs.headdim, self.configs.d_state,
            device=self.in_proj.weight.device, dtype=ssm_dtype
        )

        self.sl_offset = 0

        # Note: The Swish function with β=1 is equivalent nn.SiLU()
        self.activation = nn.SiLU() if self.configs.activation == "silu" else nn.SiLU() # TODO: Write this better

        self.chunk_size = self.configs.chunk_size
        self.use_mem_eff_path = self.configs.use_mem_eff_path
        
        self.D_has_hdim = self.configs.D_has_hdim
        self.rmsnorm = self.configs.rmsnorm
        self.norm_before_gate = self.configs.norm_before_gate
        self.dt_limit = self.configs.dt_limit

        # _init_parameters
        if self.configs.conv_init is not None:
            nn.init.uniform_(
                self.conv1d.weight, -self.configs.conv_init, self.configs.conv_init
            )

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, device=self.configs.device, dtype=self.configs.dtype)
            * (math.log(self.configs.dt_max) - math.log(self.configs.dt_min))
            + math.log(self.configs.dt_min)
        )
        dt = torch.clamp(dt, min=self.configs.dt_init_floor)

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert self.configs.A_init_range[0] > 0 and self.configs.A_init_range[1] >= self.configs.A_init_range[0]
        A = torch.empty(
            self.nheads, dtype=torch.float32, device=self.configs.device
        ).uniform_(*self.configs.A_init_range)
        A_log = torch.log(A).to(dtype=self.configs.dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(
                self.d_ssm if self.configs.D_has_hdim else self.nheads,
                device=self.configs.device,
            )
        )
        self.D._no_weight_decay = True

    def forward(
        self,
        inputs: torch.Tensor,
        sl: int = None,
        seq_idx: int = None, 
        cu_sl: int = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        inputs: (bsz, sl, hdim) if sl=None.
            If sl is not None, u is (batch * sl, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * sl dimension
            (in case batch is small).
        Returns: same shape as u
        """
        bsz, sl, _ = inputs.shape

        if self.sl_offset > 0:
            # The states are updated in-place
            preds, _, _ = self.step(inputs, self.conv_state, self.ssm_state)
            return preds

        zxbcdt = self.in_proj(inputs) # (B, L, d_in_proj) or (B * L, d_in_proj)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = (
            {}
            if self.configs.dt_limit == (0.0, float("inf"))
            else {"dt_limit": self.configs.dt_limit}
        )

        if self.configs.use_mem_eff_path:
            preds = self._forward_mem_efficient(zxbcdt, A, sl, seq_idx, dt_limit_kwargs)
        else:
            preds = self._forward_standard(
                bsz=bsz,
                zxbcdt=zxbcdt,
                A=A,
                seq_idx=seq_idx,
                dt_limit_kwargs=dt_limit_kwargs,
                cu_sl=cu_sl,
            )

        return preds

    def _forward_mem_efficient(
        self, zxbcdt: torch.Tensor, A: torch.Tensor, sl: int, seq_idx: int, dt_limit_kwargs: dict
    ) -> torch.Tensor:
        preds = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,

            D=rearrange(self.D, "(h p) -> h p", p=self.configs.headdim)
            if self.configs.D_has_hdim
            else self.D,

            chunk_size=self.configs.chunk_size,
            seq_idx=seq_idx,
            activation=self.configs.activation,
            rmsnorm_weight=self.norm.weight if self.configs.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.configs.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.configs.D_has_hdim else self.configs.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.configs.norm_before_gate,
            **dt_limit_kwargs,
        )

        if self.configs.process_group is not None:
            reduce_fn = reduce_scatter if self.configs.sequence_parallel else all_reduce
            preds = reduce_fn(preds, self.configs.process_group)

        return preds

    def _forward_standard(
        self,
        bsz: int, 
        zxbcdt: torch.Tensor,
        A: torch.Tensor,
        seq_idx: int,
        dt_limit_kwargs: dict,
        cu_sl: int = None,
    ) -> torch.Tensor:
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.configs.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.configs.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # If we just take xBC[:, :, -self.d_conv :], it will error if sl < self.d_conv
        # Instead F.pad will pad with zeros if sl < self.d_conv, and truncate otherwise.
        if cu_sl is None:
            xBC_t = rearrange(xBC, "b l d -> b d l")
            self.conv_state.copy_(F.pad(xBC_t, (self.configs.d_conv - xBC_t.shape[-1], 0))) # Update state (B D W)
        else:
            assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
            assert bsz == 1, "varlen inference only supports batch dimension 1"
            conv_varlen_states = causal_conv1d_varlen_states(
                xBC.squeeze(0), cu_sl, state_len=self.conv_state.shape[-1]
            )
            self.conv_state.copy_(conv_varlen_states)

        xBC = self._apply_conv(xBC, seq_idx)

        x, B, C = torch.split(
            xBC,
            [
                self.d_ssm,
                self.ngroups * self.configs.d_state,
                self.ngroups * self.configs.d_state,
            ],
            dim=-1,
        )

        y = self._apply_ssm(x, dt, A, B, C, seq_idx, z, dt_limit_kwargs, cu_sl=None)

        if self.configs.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        preds = self.out_proj(y)

        return preds

    def _apply_conv(self, xBC, seq_idx: torch.Tensor) -> torch.Tensor:
        if causal_conv1d_fn is None:
            assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
            return self.activation(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.d_conv - 1):]
            ) # (B, L, self.d_ssm + 2 * ngroups * d_state)
        else:
            return causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.configs.activation,
            ).transpose(1, 2)

    def _apply_ssm(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        seq_idx: int,
        z: torch.Tensor,
        dt_limit_kwargs: dict,
        cu_sl: int = None,
    ) -> torch.Tensor:
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.configs.headdim),
    
            dt,
            A,
    
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),

            chunk_size=self.configs.chunk_size,
    
            D=rearrange(self.D, "(h p) -> h p", p=self.configs.headdim)
            if self.configs.D_has_hdim
            else self.D,

            z=rearrange(z, "b l (h p) -> b l h p", p=self.configs.headdim)
            if not self.configs.rmsnorm
            else None,

            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_sl=cu_sl,
            **dt_limit_kwargs,
            return_final_states=self.ssm_state is not None,
            return_varlen_states=cu_sl is not None and (self.conv_state is not None and self.ssm_state is not None)
        )
        if self.ssm_state is not None:
            y, last_state, *rest = y
            if cu_sl is None:
                self.ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                self.ssm_state.copy_(varlen_states)
        return rearrange(y, "b l h p -> b l (h p)")

    def step(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        assert (hidden_states.shape[1] == 1), "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1)) # (B 2D)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.configs.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.configs.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        xBC = self._step_conv(xBC, dtype)

        x, B, C = torch.split(
            xBC,
            [
                self.d_ssm,
                self.ngroups * self.configs.d_state,
                self.ngroups * self.configs.d_state,
            ],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float()) # (nheads,)

        # SSM Step
        y = self._step_ssm(x, dt, A, B, C, z, dtype)

        if self.configs.rmsnorm:
            y = self.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        preds =self.out_proj(y)
        return preds.unsqueeze(1), self.conv_state, self.ssm_state

    def _step_conv(self, xBC: torch.Tensor, dtype) -> torch.Tensor:
        if causal_conv1d_update is None:
            self.conv_state.copy_(torch.roll(self.conv_state, shifts=-1, dims=-1)) # Update state (B D W)
            self.conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                self.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            ) # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.activation(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                self.conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.configs.activation,
            )
        return xBC

    def _step_ssm(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        z: torch.Tensor,
        dtype
    ) -> torch.Tensor:
        if selective_state_update is None:
            return self._step_ssm_default(x, dt, A, B, C, z, dtype)
        else:
            return self._step_ssm_selective(x, dt, A, B, C, z)

    def _step_ssm_default(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        z: torch.Tensor,
        dtype
    ) -> torch.Tensor:
        assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"

        # Discretize A and B
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype)) # (bsz, nheads)
        dA = torch.exp(dt * A) # (bsz, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.configs.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        self.ssm_state.copy_(self.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", self.ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        if not self.configs.rmsnorm:
            y = y * self.activation(z) # (B D)
        return y

    def _step_ssm_selective(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        A = repeat(A, "h -> h p n", p=self.configs.headdim, n=self.configs.d_state).to(
            dtype=torch.float32
        )
        dt = repeat(dt, "b h -> b h p", p=self.configs.headdim)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.configs.headdim)
        D = repeat(self.D, "h -> h p", p=self.configs.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.configs.headdim)
        if not self.configs.rmsnorm:
            z = rearrange(z, "b (h p) -> b h p", p=self.configs.headdim)
        y = selective_state_update(
            self.ssm_state,
            x_reshaped,
            dt,
            A,
            B,
            C,
            D,
            z=z if not self.configs.rmsnorm else None,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        return rearrange(y, "b h p -> b (h p)")

    def _initialize_states(self, bsz: int, dtype: Optional[torch.dtype] = None) -> tuple[torch.Tensor, torch.Tensor]:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        
        conv_state = torch.zeros(
            bsz, self.configs.d_conv, self.conv1d.weight.shape[0],
            device=self.conv1d.weight.device, dtype=conv_dtype
        ).transpose(1, 2)
        
        ssm_state = torch.zeros(
            bsz, self.nheads, self.configs.headdim, self.configs.d_state,
            device=self.in_proj.weight.device, dtype=ssm_dtype
        )

        return conv_state, ssm_state
