# =============================================================================#
# Authors: Tri Dao, Albert Gu, Windsor Nguyen
# File: (Mamba) model.py
# Adapted from
# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
# =============================================================================#

import math
from typing import Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)


@dataclass
class Mamba2Config:
    d_model: int
    d_state: int = 128
    d_conv: int = 4
    conv_init: Optional[float] = None
    expand: int = 2
    headdim: int = 64
    d_ssm: Optional[int] = None
    ngroups: int = 1
    A_init_range: Tuple[int, int] = (1, 16)
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: Tuple[float, float] = (0.0, float("inf"))
    bias: bool = False
    conv_bias: bool = True
    chunk_size: int = 256
    use_mem_eff_path: bool = True
    layer_idx: Optional[int] = None
    process_group: Optional[Any] = None
    sequence_parallel: bool = True
    device: Optional[Any] = None
    dtype: Optional[Any] = None
    dropout: float = 0.25
    loss_fn: Optional[Any] = None
    max_len: int = 1000
    d_out: int = 29


@dataclass
class InferenceParameters:
    key_value_memory_dict: dict = field(default_factory=dict)
    seqlen_offset: int = 0


class Mamba2(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.factory_kwargs = {"device": config.device, "dtype": config.dtype}

        self._init_dimensions()
        self._init_layers()
        self._init_parameters()

    def _init_dimensions(self):
        self.d_inner = (self.config.expand * self.config.d_model) // self.world_size
        assert (
            self.d_inner * self.world_size == self.config.expand * self.config.d_model
        )

        self.d_ssm = (
            self.d_inner
            if self.config.d_ssm is None
            else self.config.d_ssm // self.world_size
        )
        assert self.config.ngroups % self.world_size == 0

        self.ngroups = self.config.ngroups // self.world_size
        assert self.d_ssm % self.config.headdim == 0
        self.nheads = self.d_ssm // self.config.headdim

    def _init_layers(self):
        d_in_proj = (
            2 * self.d_inner + 2 * self.ngroups * self.config.d_state + self.nheads
        )

        if self.config.process_group is None:
            self.in_proj = nn.Linear(
                self.config.d_model,
                d_in_proj,
                bias=self.config.bias,
                **self.factory_kwargs,
            )
        else:
            self.in_proj = ColumnParallelLinear(
                self.config.d_model,
                d_in_proj * self.world_size,
                bias=self.config.bias,
                process_group=self.config.process_group,
                sequence_parallel=self.config.sequence_parallel,
                **self.factory_kwargs,
            )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.config.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=self.config.conv_bias,
            kernel_size=self.config.d_conv,
            groups=conv_dim,
            padding=self.config.d_conv - 1,
            **self.factory_kwargs,
        )

        if self.config.rmsnorm:
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.config.norm_before_gate,
                group_size=self.d_ssm // self.config.ngroups,
                **self.factory_kwargs,
            )

        if self.config.process_group is None:
            self.out_proj = nn.Linear(
                self.d_inner,
                self.config.d_out,
                bias=self.config.bias,
                **self.factory_kwargs,
            )
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size,
                self.config.d_out,
                bias=self.config.bias,
                process_group=self.config.process_group,
                sequence_parallel=self.config.sequence_parallel,
                **self.factory_kwargs,
            )

    def _init_parameters(self):
        if self.config.conv_init is not None:
            nn.init.uniform_(
                self.conv1d.weight, -self.config.conv_init, self.config.conv_init
            )

        dt = torch.exp(
            torch.rand(self.nheads, **self.factory_kwargs)
            * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        )
        dt = torch.clamp(dt, min=self.config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.empty(
            self.nheads, dtype=torch.float32, device=self.config.device
        ).uniform_(*self.config.A_init_range)
        A_log = torch.log(A).to(dtype=self.config.dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(
            torch.ones(
                self.d_ssm if self.config.D_has_hdim else self.nheads,
                device=self.config.device,
            )
        )
        self.D._no_weight_decay = True

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParameters] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = inputs.shape

        u = inputs

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(
                inference_params, batch_size
            )
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out, None

        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)
        dt_limit_kwargs = (
            {}
            if self.config.dt_limit == (0.0, float("inf"))
            else {"dt_limit": self.config.dt_limit}
        )

        if self.config.use_mem_eff_path and inference_params is None:
            out = self._forward_mem_efficient(zxbcdt, A, dt_limit_kwargs)
        else:
            out = self._forward_standard(
                zxbcdt, A, dt_limit_kwargs, conv_state, ssm_state
            )

        loss = self.config.loss_fn(out, targets) if targets is not None else None
        return out, loss

    def _forward_mem_efficient(
        self, zxbcdt: torch.Tensor, A: torch.Tensor, dt_limit_kwargs: dict
    ) -> torch.Tensor:
        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=rearrange(self.D, "(h p) -> h p", p=self.config.headdim)
            if self.config.D_has_hdim
            else self.D,
            chunk_size=self.config.chunk_size,
            seq_idx=None,
            activation=self.config.activation,
            rmsnorm_weight=self.norm.weight if self.config.rmsnorm else None,
            rmsnorm_eps=self.norm.eps if self.config.rmsnorm else 1e-6,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=None if self.config.D_has_hdim else self.config.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.config.norm_before_gate,
            **dt_limit_kwargs,
        )
        if self.config.process_group is not None:
            reduce_fn = reduce_scatter if self.config.sequence_parallel else all_reduce
            out = reduce_fn(out, self.config.process_group)
        return out

    def _forward_standard(
        self,
        zxbcdt: torch.Tensor,
        A: torch.Tensor,
        dt_limit_kwargs: dict,
        conv_state: Optional[torch.Tensor],
        ssm_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.config.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.config.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        if conv_state is not None:
            xBC_t = rearrange(xBC, "b l d -> b d l")
            conv_state.copy_(F.pad(xBC_t, (self.config.d_conv - xBC_t.shape[-1], 0)))

        xBC = self._apply_conv(xBC)
        x, B, C = torch.split(
            xBC,
            [
                self.d_ssm,
                self.ngroups * self.config.d_state,
                self.ngroups * self.config.d_state,
            ],
            dim=-1,
        )

        y = self._apply_ssm(x, dt, A, B, C, ssm_state, z, dt_limit_kwargs)

        if self.config.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)

        return out

    def _apply_conv(self, xBC: torch.Tensor) -> torch.Tensor:
        if causal_conv1d_fn is None or self.config.activation not in ["silu", "swish"]:
            return F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        else:
            return causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.config.activation,
            ).transpose(1, 2)

    def _apply_ssm(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        ssm_state: Optional[torch.Tensor],
        z: torch.Tensor,
        dt_limit_kwargs: dict,
    ) -> torch.Tensor:
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.config.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.config.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.config.headdim)
            if self.config.D_has_hdim
            else self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.config.headdim)
            if not self.config.rmsnorm
            else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=None,
            **dt_limit_kwargs,
            return_final_states=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        return rearrange(y, "b l h p -> b l (h p)")

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.config.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.config.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        xBC = self._step_conv(xBC, conv_state)
        x, B, C = torch.split(
            xBC,
            [
                self.d_ssm,
                self.ngroups * self.config.d_state,
                self.ngroups * self.config.d_state,
            ],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())

        y = self._step_ssm(x, dt, A, B, C, ssm_state, z)

        if self.config.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def _step_conv(self, xBC: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = F.silu(xBC).to(dtype=xBC.dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.config.activation,
            )
        return xBC

    def _step_ssm(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        ssm_state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if selective_state_update is None:
            return self._step_ssm_default(x, dt, A, B, C, ssm_state, z)
        else:
            return self._step_ssm_selective(x, dt, A, B, C, ssm_state, z)

    def _step_ssm_default(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        ssm_state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
        dA = torch.exp(dt * A)
        x = rearrange(x, "b (h p) -> b h p", p=self.config.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        if not self.config.rmsnorm:
            y = y * F.silu(z)
        return y

    def _step_ssm_selective(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        ssm_state: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        A = repeat(A, "h -> h p n", p=self.config.headdim, n=self.config.d_state).to(
            dtype=torch.float32
        )
        dt = repeat(dt, "b h -> b h p", p=self.config.headdim)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.config.headdim)
        D = repeat(self.D, "h -> h p", p=self.config.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.config.headdim)
        if not self.config.rmsnorm:
            z = rearrange(z, "b (h p) -> b h p", p=self.config.headdim)
        y = selective_state_update(
            ssm_state,
            x_reshaped,
            dt,
            A,
            B,
            C,
            D,
            z=z if not self.config.rmsnorm else None,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        return rearrange(y, "b h p -> b (h p)")

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.conv1d.weight.shape[0],
            self.config.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.config.headdim,
            self.config.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self,
        inference_params: InferenceParameters,
        batch_size: int,
        initialize_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.config.layer_idx is not None
        if self.config.layer_idx not in inference_params.key_value_memory_dict:
            conv_state, ssm_state = self._initialize_states(batch_size)
            inference_params.key_value_memory_dict[self.config.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[
                self.config.layer_idx
            ]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def _initialize_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_state = torch.zeros(
            batch_size,
            self.conv1d.weight.shape[0],
            self.config.d_conv,
            device=self.conv1d.weight.device,
            dtype=self.conv1d.weight.dtype,
        )
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.config.headdim,
            self.config.d_state,
            device=self.in_proj.weight.device,
            dtype=self.in_proj.weight.dtype,
        )
        return conv_state, ssm_state
