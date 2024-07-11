# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: model.py
# ==============================================================================#

"""The Spectral State Space Model architecture."""

import math

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from models.stu.stu_utils import (
    get_top_eigh, 
    preconvolve,
    compute_ar_u, 
    compute_spectral, 
    compute_ar_y
)
from utils.nearest_power_of_2 import nearest_power_of_2
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU
from tqdm import tqdm
from torch.nn import MSELoss


@dataclass
class SpectralSSMConfigs:
    d_in: int = 37
    d_out: int = 37
    d_proj: int = 29
    n_layers: int = 2
    n_embd: int = 37
    sl: int = 1_000
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 16
    k_y: int = 2  # Number of parametrizable, autoregressive matrices Mʸ
    k_u: int = 3  # Number of parametrizable, autoregressive matrices Mᵘ
    learnable_m_y: bool = True
    alpha: float = 0.9  # 0.9 deemed "uniformly optimal" in the paper
    use_ar_y: bool = False
    use_ar_u: bool = False
    use_hankel_L: bool = False
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )
    device: torch.device = None


class STU(nn.Module):
    """
    An STU (Spectral Transform Unit) layer.

    Args:
        configs: Configuration contains (at least) the following attributes:
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            num_eigh (int): Number of spectral filters to use.
            use_ar_y (bool): Use autoregressive on the output sequence?
            use_ar_u (bool): Use autoregressive on the input sequence?
            k_u (int): Autoregressive depth on the input sequence.
            k_y (int): Autoregressive depth on the output sequence.
            learnable_m_y (bool): Learn the M_y matrix?
            dropout (float): Dropout rate.
        sigma (torch.Tensor): Eigenvalues of the Hankel matrix.
        V (torch.Tensor): Precomputed FFT of top K eigenvectors.
    """

    def __init__(self, configs, sigma, V, padded_sl) -> None:
        super(STU, self).__init__()
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.k = configs.num_eigh
        self.use_ar_y = configs.use_ar_y
        self.use_ar_u = configs.use_ar_u
        self.sigma = sigma
        self.V = V # Precomputed FFT of top K eigenvectors.
        self.padded_sl = padded_sl
        self.k_u = configs.k_u
        self.k_y = configs.k_y
        self.learnable_m_y = configs.learnable_m_y
        self.stu_dropout = nn.Dropout(configs.dropout)
        self.resid_dropout = nn.Dropout(configs.dropout)
        
        # Parameterizable matrix Mᵘ, Mᵠ⁺, and Mᵠ⁻, per section 3
        self.M_u = nn.Parameter(torch.empty(self.k_u, self.d_out, self.d_in))
        self.M_phi_plus = nn.Parameter(torch.empty(self.k, self.d_out, self.d_in))
        self.M_phi_minus = nn.Parameter(torch.empty(self.k, self.d_out, self.d_in))

        # Parametrizable matrix Mʸ Introduced in section 5, equation 5
        if self.learnable_m_y:
            self.M_y = nn.Parameter(torch.zeros(self.d_out, self.k_y, self.d_out))
        else:
            self.register_buffer("m_y", torch.zeros(self.d_out, self.k_y, self.d_out))


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STU layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (bsz, sl, d_out)
        """
        spectral = self.stu_dropout(
            compute_spectral(
                inputs, self.sigma, self.V, self.M_phi_plus, 
                self.M_phi_minus, self.M_y, self.padded_sl, self.use_ar_y
            )
        )

        match (self.use_ar_u, self.use_ar_y):
            case (True, True):
                y_t = spectral + compute_ar_u(self.M_u, inputs)
                y_t += compute_ar_y(self.M_y, y_t)
            case (True, False):
                y_t = spectral + compute_ar_u(self.M_u, inputs)
            case (False, True):
                y_t = spectral
                y_t += compute_ar_y(self.M_y, y_t)
            case (False, False):
                y_t = spectral

        return self.resid_dropout(y_t)


class MLP(nn.Module):
    """
    Multi-layer perceptron network using SwiGLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            scale (float): Scaling factor for hidden dimension.
            n_embd (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs) -> None:
        super(MLP, self).__init__()
        self.h_dim = configs.scale * configs.n_embd
        self.swiglu = SwiGLU(dim=configs.n_embd, h_dim=self.h_dim, bias=configs.bias, use_sq_relu=False)
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
        x = self.dropout(x)
        return x


class GatedMLP(nn.Module):
    """
    Gated Multi-layer perceptron network using SiLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            n_embd (int): Input and output embedding dimension.
            scale (float): Scaling factor for hidden dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super().__init__()
        self.in_features = configs.n_embd
        self.out_features = configs.n_embd
        self.hidden_features = int(configs.scale * configs.n_embd)

        self.fc1 = nn.Linear(self.in_features, 2 * self.hidden_features, bias=configs.bias)
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=configs.bias)
        self.activation = torch.nn.functional.silu
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
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return self.dropout(y)

class Block(nn.Module):
    """
    A single block of the spectral SSM model composed of STU and MLP layers.

    Args:
        configs: Configuration object for STU and MLP layers
        sigma (torch.Tensor): Eigenvalues of the Hankel matrix.
        V (torch.Tensor): Precomputed FFT of top K eigenvectors.
        padded_sl (int): Padded sequence length for FFT operations.
    """

    def __init__(self, configs, sigma, V, padded_sl) -> None:
        super(Block, self).__init__()
        self.rn_1 = RMSNorm(configs.n_embd)
        self.rn_2 = RMSNorm(configs.n_embd)
        self.stu = STU(configs, sigma, V, padded_sl)
        self.rn_3 = RMSNorm(configs.n_embd)
        self.mlp = MLP(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        z = self.rn_1(x)
        x = z + self.stu(self.rn_2(x))
        x = x + self.mlp(self.rn_3(x))
        return x + z  

class SpectralSSM(nn.Module):
    """
    Model architecture based on stacked blocks of STU and MLP layers.

    Args:
        configs: Configuration object containing model hyperparameters.
    """

    def __init__(self, configs) -> None:
        super(SpectralSSM, self).__init__()
        self.configs = configs
        self.n_layers = configs.n_layers
        self.n_embd = configs.n_embd
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.d_proj = configs.d_proj
        self.sl = configs.sl
        self.num_eigh = configs.num_eigh
        self.learnable_m_y = configs.learnable_m_y
        self.alpha = configs.alpha
        self.use_hankel_L = configs.use_hankel_L
        self.device = configs.device

        self.sigma, self.phi = get_top_eigh(self.sl, self.num_eigh, self.use_hankel_L, self.device)
        self.V, self.padded_sl = preconvolve(self.phi, self.sl) # Precomputed.

        self.bias = configs.bias
        self.dropout = configs.dropout
        self.loss_fn = configs.loss_fn
        self.controls = configs.controls
        
        self.spectral_ssm = nn.ModuleDict(
            dict(
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList(
                    [Block(
                        self.configs, self.sigma, self.V, self.padded_sl
                    ) for _ in range(self.n_layers)]),
            )
        )
        self.output = nn.Linear(self.n_embd, self.d_proj, bias=self.bias)

        # Initialize all weights
        self.m_x = self.d_out**-0.5
        self.std = self.n_embd**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print("\nSTU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, inputs, targets):
        """
        Forward pass of the spectral SSM model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor): Target tensor for loss computation

        Returns:
            Type (ignore due to high variability):
            - Predictions tensor
            - Tuple containing loss and metrics (if applicable)
        """
        _, sl, n_embd = inputs.size()
        x = self.spectral_ssm.dropout(inputs)
        for block in self.spectral_ssm.hidden:
            x = block(x)
        preds = self.output(x)

        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            return preds, loss

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            torch.nn.init.uniform_(module.M_u, -self.m_x, self.m_x)
            torch.nn.init.xavier_normal_(module.M_phi_plus)
            torch.nn.init.xavier_normal_(module.M_phi_minus)

            # Initialize Mʸ₂ = α * I, page 8.
            if self.learnable_m_y and module.k_y > 1:
                with torch.no_grad():
                    module.M_y[:, 1] = self.alpha * torch.eye(module.d_out)

    def get_num_params(self):
        """
        Return the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
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
        L, D, E, T = cfg.n_layers, cfg.n_embd, cfg.num_eigh, cfg.sl

        total_flops = 0

        # STU blocks
        for _ in range(L):
            total_flops += self._compute_rmsnorm_flops(D, T) * 2  # 2 RMSNorm per block
            total_flops += self._compute_spectral_flops(T, D, E)
            
            if cfg.use_ar_u:
                total_flops += 2 * cfg.k_u * D * D * T  # compute_ar_u
            if cfg.use_ar_y:
                total_flops += self._compute_ar_y_flops(T, D, cfg.k_y)

            total_flops += self._compute_swiglu_flops(D, cfg.scale, T)

        # Output layer
        total_flops += 2 * D * cfg.d_proj * T

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

    def _compute_spectral_flops(self, T: int, D: int, E: int) -> int:
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
        assert rollout_steps <= steps, f"Cannot roll out for more than total steps"

        # Track model hallucinations
        predicted_steps = torch.zeros(num_traj, steps, d_out, device=device)

        # Track loss between rollout vs ground truth
        traj_losses = torch.zeros(num_traj, steps, device=device)

        # Initialize cost function
        mse_loss = MSELoss()

        for step in tqdm(range(steps), desc="Predicting", unit="step"):
            current_step = init + step

            # Predict the next state using a fixed window size of inputs
            step_preds, (_, _) = self.forward(
                inputs[:, :current_step], targets[:, :current_step]
            )

            # Calculate the mean loss of the last rollout_steps predictions
            rollout_preds = step_preds[:, -rollout_steps:, :]
            rollout_ground_truths = targets[:, (current_step + 1 - rollout_steps) : (current_step + 1), :]
            traj_losses[:, step] = mse_loss(rollout_preds, rollout_ground_truths)

            # Store the last prediction step for plotting
            predicted_steps[:, step] = step_preds[:, -1].squeeze(1)

            # # Concatenate the autoregressive predictions of states and the ground truth actions (hallucinating)
            # next_action = inputs[:, current_step:current_step+1, -(d_in - d_out):]
            # next_input = torch.cat([next_input, next_action], dim=2)
            # ar_inputs = torch.cat([ar_inputs, next_input], dim=1)

        avg_loss = traj_losses.mean()
        return predicted_steps, (avg_loss, traj_losses)
