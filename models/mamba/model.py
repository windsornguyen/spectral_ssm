# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: model.py
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
    d_in: int = 29
    d_model: int = 32
    d_out: int = 29
    mlp_scale: int = 4
    embd_scale: int = 1
    dropout: float = 0.10
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

    # MoE
    moe: bool = True
    num_experts: int = 8
    num_experts_per_timestep: int = 2

    # TODO: Experiment-specific hyperparameters
    loss_fn: Optional[any] = nn.SiLU()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )


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
        self.controls = self.configs.controls

        if configs.moe:
            print(f"\nMoE?: Enabled | Using {configs.num_experts} experts.")
        else:
            print("\nMoE?: Disabled")
            
        self.input_proj = nn.Linear(self.d_in, self.d_model * self.embd_scale, bias=self.bias)

        self.mamba = nn.ModuleDict(
            dict(
                hidden=nn.ModuleList(
                    [MambaBlock(self.configs) for _ in range(self.n_layers)]),
            )
        )
        self.output_proj = nn.Linear(self.d_model * self.embd_scale, self.d_out, bias=self.bias)

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
        Forward pass of the spectral SSM model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor): Target tensor for loss computation

        Returns:
            Type (ignore due to high variability):
            - Predictions tensor
            - tuple containing loss and metrics (if applicable)
        """
        x = self.input_proj(inputs)
        for block in self.mamba.hidden:
            x = block(x)
        preds = self.output_proj(x)

        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            return preds, loss
    
    # TODO: Incorrectly written for Mamba-2
    def estimate_mfu(self, fwdbwd_per_iter: float, dt: float) -> tuple[float, float]:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        Args:
            fwdbwd_per_iter (float): Number of forward/backward passes per iteration.
            dt (float): Time taken for the iteration.

        Returns:
            tuple[float, float]: Estimated MFU and estimated peak FLOPS.

        Reference:
            PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        """
        cfg = self.configs
        L, D, E, T = cfg.n_layers, cfg.d_model, cfg.num_eigh, cfg.sl

        total_flops = 0

        # Mamba2 blocks
        for _ in range(L):
            total_flops += self._compute_rmsnorm_flops(D, T) * 2  # 2 RMSNorm per block
            total_flops += self._compute_mamba_flops(T, D, E)
            
            if cfg.use_ar_u:
                total_flops += 2 * cfg.k_u * D * D * T  # compute_ar_u
            if cfg.use_ar_y:
                total_flops += self._compute_ar_y_flops(T, D, cfg.k_y)

            total_flops += self._compute_swiglu_flops(D, cfg.scale, T)

        # Output layer
        total_flops += 2 * D * cfg.d_out * T

        # Dropout operations (1 FLOP per element)
        total_flops += (L + 1) * D * T  # L blocks + initial dropout

        flops_per_iter = total_flops
        flops_achieved = flops_per_iter * fwdbwd_per_iter / dt
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops (312 TFLOPS)
        return flops_achieved, flops_promised

    def _compute_rmsnorm_flops(self, D: int, T: int) -> int:
        """
        Compute FLOPS for RMSNorm operation.

        Args:
            D (int): Embedding dimension.
            T (int): Sequence length.

        Returns:
            int: Total FLOPS for RMSNorm.
        """
        power_flops = D * T  # x^2
        sum_flops = D * T  # Sum across dimension
        mean_flops = T  # Division for mean
        sqrt_flops = T  # Square root
        normalize_flops = D * T  # Division by sqrt(mean + eps)
        scale_flops = D * T  # Multiplication by weight

        return power_flops + sum_flops + mean_flops + sqrt_flops + normalize_flops + scale_flops

    def _compute_mamba_flops(self, T: int, D: int, E: int) -> int:
        """
        Compute FLOPS for spectral computations.

        Args:
            T (int): Sequence length.
            D (int): Embedding dimension.
            E (int): Number of eigenvectors.

        Returns:
            int: Total FLOPS for spectral computations.
        """
        n = nearest_power_of_2(T * 2 - 1)
        log_n = math.log2(n)

        fft_flops = 2 * n * log_n * D * E * 2  # rfft and irfft
        filter_flops = 2 * T * D * E  # Spectral filter application

        return int(fft_flops + filter_flops)

    def _compute_ar_y_flops(self, T: int, D: int, k_y: int) -> int:
        """
        Compute FLOPS for compute_ar_y function.

        Args:
            T (int): Sequence length.
            D (int): Embedding dimension.
            k_y (int): Autoregressive depth on the output sequence.

        Returns:
            int: Total FLOPS for ar_y computation.
        """
        return 2 * T * k_y * D * D + T * k_y * D  # Matrix multiplications + additions

    def _compute_swiglu_flops(self, D: int, scale: float, T: int) -> int:
        """
        Compute FLOPS for SwiGLU activation.

        Args:
            D (int): Embedding dimension.
            scale (float): Scaling factor for hidden dimension.
            T (int): Sequence length.

        Returns:
            int: Total FLOPS for SwiGLU computation.
        """
        h_dim = int(scale * D)
        
        linear_flops = 2 * D * h_dim * T * 3  # Three linear layers
        silu_flops = 3 * h_dim * T  # SiLU activation (3 ops per element)
        hadamard_flops = h_dim * T  # Hadamard product
        
        return linear_flops + silu_flops + hadamard_flops

    def predict_states(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 950,
        steps: int = 50,
        rollout_steps: int = 20,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]
        ],
    ]:
        """
        Perform autoregressive prediction with optional periodic grounding to true targets.

        Args:
            inputs (torch.Tensor): Input tensor of shape (num_traj, total_steps, d_in)
            targets (torch.Tensor): Target tensor of shape (num_traj, total_steps, d_out)
            init (int): Index of the initial state to start the prediction
            steps (int): Number of steps to predict

        Returns:
        tuple: Contains the following elements:
            - preds (torch.Tensor): Predictions of shape (num_traj, total_steps, d_out)
            - tuple:
                - avg_loss (torch.Tensor): Scalar tensor with the average loss
                - traj_losses (torch.Tensor): Losses for each trajectory and step, shape (num_traj, steps)
        """
        device = next(self.parameters()).device
        print(f"Predicting on {device}.")
        num_traj, total_steps, d_out = targets.size()
        assert init + steps <= total_steps, f"Cannot take more steps than {total_steps}"
        assert rollout_steps <= steps, "Cannot roll out for more than total steps"

        # Track model hallucinations
        predicted_steps = torch.zeros(num_traj, steps, d_out, device=device)

        # Track loss between rollout vs ground truth
        traj_losses = torch.zeros(num_traj, steps, device=device)

        # Initialize cost function
        mse_loss = nn.MSELoss()

        for step in tqdm(range(steps), desc="Predicting", unit="step"):
            current_step = init + step

            # Predict the next state using a fixed window size of inputs
            step_preds, (_, _) = self.forward(
                inputs[:, :current_step], targets[:, :current_step]
            )

            # Calculate the mean loss of the last rollout_steps predictions
            rollout_preds = step_preds[:, -rollout_steps:, :]
            rollout_ground_truths = targets[:, (current_step - rollout_steps) : current_step, :]
            traj_losses[:, step] = mse_loss(rollout_preds, rollout_ground_truths)

            # Store the last prediction step for plotting
            predicted_steps[:, step] = step_preds[:, -1].squeeze(1)

            # # Concatenate the autoregressive predictions of states and the ground truth actions (hallucinating)
            # next_action = inputs[:, current_step:current_step+1, -(d_in - d_out):]
            # next_input = torch.cat([next_input, next_action], dim=2)
            # ar_inputs = torch.cat([ar_inputs, next_input], dim=1)

        avg_loss = traj_losses.mean()
        return predicted_steps, (avg_loss, traj_losses)
