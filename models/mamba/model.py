# =============================================================================#
# Authors: Tri Dao, Albert Gu, Windsor Nguyen, Isabel Liu
# File: (Mamba-2) model.py
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
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

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
class Mamba2Configs:
    d_model: int
    d_state: int = 128
    d_conv: int = 4
    conv_init: Optional[float] = None
    expand: int = 2 # The paper sets the expand factor E = 2
    headdim: int = 64
    d_ssm: Optional[int] = None # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
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

    # Fused kernel and sharding options
    chunk_size: int = 256
    use_mem_eff_path: bool = True
    layer_idx: Optional[int] = None # Absorb kwarg for general module
    process_group: Optional[Any] = None
    sequence_parallel: bool = True
    device: Optional[Any] = None
    dtype: Optional[Any] = None

    # TODO: Experiment-specific hyperparameters
    dropout: float = 0.10
    loss_fn: Optional[Any] = None
    sl: int = 900
    d_out: int = 29


@dataclass
class InferenceParameters:
    kv_mem_dict: dict = field(default_factory=dict)
    seqlen_offset: int = 0


class Mamba2(nn.Module):
    def __init__(self, configs: Mamba2Configs):
        super().__init__()
        self.configs = configs
        self.factory_kwargs = {"device": configs.device, "dtype": configs.dtype}

        self.ngroups = self.configs.ngroups // self.world_size
        assert self.d_ssm % self.configs.headdim == 0

        self.nheads = self.d_ssm // self.configs.headdim
        self.D_has_hdim = self.configs.D_has_hdim
        self.rmsnorm = self.configs.rmsnorm # TODO <-- May have to adjust the Triton path in configs
        self.norm_before_gate = self.configs.norm_before_gate
        self.dt_limit = dt_limit

        # Note: The Swish function with β=1 is equivalent nn.SiLU()
        self.activation = nn.SiLU()

        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self._init_dimensions()
        self._init_layers()
        self._init_parameters()

    def _init_dimensions(self):
        self.d_inner = (self.configs.expand * self.configs.d_model) // self.world_size
        assert (self.d_inner * self.world_size == self.configs.expand * self.configs.d_model)

        self.d_ssm = (
            self.d_inner
            if self.configs.d_ssm is None
            else self.configs.d_ssm // self.world_size
        )
        assert self.configs.ngroups % self.world_size == 0

    def _init_layers(self):
        d_in_proj = (2 * self.d_inner + 2 * self.ngroups * self.configs.d_state + self.nheads)

        if self.configs.process_group is None:
            self.in_proj = nn.Linear(
                self.configs.d_model,
                d_in_proj,
                bias=self.configs.bias,
                **self.factory_kwargs,
            )
        else:
            self.in_proj = ColumnParallelLinear(
                self.configs.d_model,
                d_in_proj * self.world_size,
                bias=self.configs.bias,
                process_group=self.configs.process_group,
                sequence_parallel=self.configs.sequence_parallel,
                **self.factory_kwargs,
            )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.configs.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=self.configs.conv_bias,
            kernel_size=self.configs.d_conv,
            groups=conv_dim,
            padding=self.configs.d_conv - 1,
            **self.factory_kwargs,
        )

        if self.configs.rmsnorm:
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.configs.norm_before_gate,
                group_size=self.d_ssm // self.configs.ngroups,
                **self.factory_kwargs,
            )

        if self.configs.process_group is None:
            self.out_proj = nn.Linear(
                self.d_inner,
                self.configs.d_out,
                bias=self.configs.bias,
                **self.factory_kwargs,
            )
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size,
                self.configs.d_out,
                bias=self.configs.bias,
                process_group=self.configs.process_group,
                sequence_parallel=self.configs.sequence_parallel,
                **self.factory_kwargs,
            )

    def _init_parameters(self):
        if self.configs.conv_init is not None:
            nn.init.uniform_(
                self.conv1d.weight, -self.configs.conv_init, self.configs.conv_init
            )

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **self.factory_kwargs)
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
        targets: Optional[torch.Tensor] = None,
        sl: int = None,
        seq_idx=None, 
        cu_sl=None,
        inference_params: Optional[InferenceParameters] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        u: (bsz, sl, hdim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        sl_og = sl
        if sl is None:
            bsz, sl, _ = inputs.shape
        else:
            bsz_sl, _ = inputs.shape
            bsz = bsz_sl // sl

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_sl.shape[0] - 1 if cu_sl is not None else bsz
            conv_state, ssm_state = self._get_states_from_cache(
                inference_params, inference_batch
            )
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u) # (B, L, d_in_proj) or (B * L, d_in_proj)
        if sl_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=sl)

        # TODO: If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = (
            {}
            if self.configs.dt_limit == (0.0, float("inf"))
            else {"dt_limit": self.configs.dt_limit}
        )

        if self.configs.use_mem_eff_path and inference_params is None:
            out = self._forward_mem_efficient(zxbcdt, A, sl_og, seq_idx, dt_limit_kwargs)
        else:
            out = self._forward_standard(
                bsz=bsz,
                zxbcdt=zxbcdt,
                A=A,
                cu_sl=cu_sl,
                seq_idx=seq_idx,
                dt_limit_kwargs=dt_limit_kwargs,
                conv_state=conv_state,
                ssm_state=ssm_state
            )

        loss = self.configs.loss_fn(out, targets) if targets is not None else None
        return out, loss

    def _forward_mem_efficient(
        self, zxbcdt: torch.Tensor, A: torch.Tensor, sl_og = None, seq_idx=None, dt_limit_kwargs: dict
    ) -> torch.Tensor:
        out = mamba_split_conv1d_scan_combined(
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
        if sl_og is not None:
            out = rearrange(out, "b l d -> (b l) d")
        if self.configs.process_group is not None:
            reduce_fn = reduce_scatter if self.configs.sequence_parallel else all_reduce
            out = reduce_fn(out, self.configs.process_group)
        return out

    def _forward_standard(
        self,
        bsz: int, 
        zxbcdt: torch.Tensor,
        A: torch.Tensor,
        cu_sl: int,
        seq_idx: int,
        dt_limit_kwargs: dict,
        conv_state: Optional[torch.Tensor],
        ssm_state: Optional[torch.Tensor],
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

        if conv_state is not None:
            if cu_sl is None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t, (self.configs.d_conv - xBC_t.shape[-1], 0)))
        else:
            assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
            assert bsz == 1, "varlen inference only supports batch dimension 1"
            conv_varlen_states = causal_conv1d_varlen_states(
                xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
            )
            conv_state.copy_(conv_varlen_states)

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

        y = self._apply_ssm(x, dt, A, B, C, seq_idx, cu_sl, ssm_state, z, dt_limit_kwargs)

        if self.configs.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)

        return out

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
        cu_sl: int,
        ssm_state: Optional[torch.Tensor],
        z: torch.Tensor,
        dt_limit_kwargs: dict,
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
            return_final_states=ssm_state is not None,
            return_varlen_states=cu_sl is not None and inference_params is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            if cu_sl is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)
        return rearrange(y, "b l h p -> b l (h p)")

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        xBC = self._step_conv(xBC, conv_state, dtype)

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
        y = self._step_ssm(x, dt, A, B, C, ssm_state, z)

        if self.configs.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def _step_conv(self, xBC: torch.Tensor, conv_state: torch.Tensor, dtype) -> torch.Tensor:
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1)) # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            ) # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.activation(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
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

        # Discretize A and B
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype)) # (bsz, nheads)
        dA = torch.exp(dt * A) # (bsz, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.configs.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
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
        ssm_state: torch.Tensor,
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
            ssm_state,
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

    def allocate_inference_cache(
        self, bsz: int, max_seqlen: int, dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            bsz,
            self.conv1d.weight.shape[0],
            self.configs.d_conv,
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            bsz,
            self.nheads,
            self.configs.headdim,
            self.configs.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self,
        inference_params: InferenceParameters,
        bsz: int,
        initialize_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.configs.layer_idx is not None
        if self.configs.layer_idx not in inference_params.kv_mem_dict:
            conv_state, ssm_state = self._initialize_states(bsz)
            inference_params.kv_mem_dict[self.configs.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = inference_params.kv_mem_dict[
                self.configs.layer_idx
            ]
            # TODO: What if bsz changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def _initialize_states(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_state = torch.zeros(
            bsz,
            self.conv1d.weight.shape[0],
            self.configs.d_conv,
            device=self.conv1d.weight.device,
            dtype=self.conv1d.weight.dtype,
        ).transpose(1, 2)

        ssm_state = torch.zeros(
            bsz,
            self.nheads,
            self.configs.headdim,
            self.configs.d_state,
            device=self.in_proj.weight.device,
            dtype=self.in_proj.weight.dtype,
        )
        return conv_state, ssm_state
