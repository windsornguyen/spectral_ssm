# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: model.py
# ==============================================================================#

"""The Spectral State Space Model architecture."""

import math
import os
import csv
import time
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from models.stu.stu_utils import (
    get_top_eigh,
    preconvolve,
    compute_ar_u,
    compute_spectral,
    compute_ar_y,
)
from utils.nearest_power_of_2 import nearest_power_of_2
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU
from tqdm import tqdm


@dataclass
class SpectralSSMConfigs:
    d_in: int = 37
    d_out: int = 29
    n_layers: int = 4
    d_model: int = 37
    sl: int = 1_000
    mlp_scale: int = 4
    embd_scale: int = 1
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
    # num_stu_mlp_pairs: int = 3
    num_models: int = 3
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )
    device: torch.device = None

    # MoE
    moe: bool = True
    num_experts: int = 8
    num_experts_per_timestep: int = 2


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
        self.d_model = configs.d_model
        self.embd_scale = configs.embd_scale
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
        self.M_u = nn.Parameter(torch.empty(self.k_u, (self.d_model * self.embd_scale), (self.d_model * self.embd_scale)))
        self.M_phi_plus = nn.Parameter(torch.empty(self.k, (self.d_model * self.embd_scale), (self.d_model * self.embd_scale)))
        self.M_phi_minus = nn.Parameter(torch.empty(self.k, (self.d_model * self.embd_scale), (self.d_model * self.embd_scale)))

        # Parametrizable matrix Mʸ Introduced in section 5, equation 5
        if self.learnable_m_y:
            self.M_y = nn.Parameter(torch.zeros((self.d_model * self.embd_scale), self.k_y, (self.d_model * self.embd_scale)))
        else:
            self.register_buffer("m_y", torch.zeros((self.d_model * self.embd_scale), self.k_y, (self.d_model * self.embd_scale)))


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STU layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (bsz, sl, d_model)
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
            d_model (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs) -> None:
        super(MLP, self).__init__()
        self.h_dim = int(configs.mlp_scale * configs.d_model)
        self.swiglu = SwiGLU(dim=configs.d_model * configs.embd_scale, h_dim=self.h_dim, bias=configs.bias, use_sq_relu=False)
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

        self.fc_1 = nn.Linear(self.in_features, self.chunks * self.hidden_features, bias=configs.bias)
        self.fc_2 = nn.Linear(self.hidden_features, self.out_features, bias=configs.bias)
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
        y = self.fc_1(x)
        y, gate = y.chunk(self.chunks, dim=-1)
        y = y * self.silu(gate)
        y = self.fc_2(y)
        return self.dropout(y)

class SimpleGateMoe(nn.Module):
    """
    A single block of the spectral SSM model composed of STU and MLP layers,
    with a gating mechanism for input-dependent selectivity.

    Args:
        configs: Configuration object for STU and MLP layers
        sigma (torch.Tensor): Eigenvalues of the Hankel matrix.
        V (torch.Tensor): Precomputed FFT of top K eigenvectors.
        padded_sl (int): Padded sequence length for FFT operations.
    """

    def __init__(self, configs, sigma, V, padded_sl) -> None:
        super(SimpleGateMoe, self).__init__()
        self.rn = RMSNorm(configs.d_model * configs.embd_scale)
        self.stu_1 = STU(configs, sigma, V, padded_sl)
        self.stu_2 = STU(configs, sigma, V, padded_sl)
        self.stu_3 = STU(configs, sigma, V, padded_sl)
        self.stu_4 = STU(configs, sigma, V, padded_sl)
        self.gate = nn.Linear(configs.d_model * configs.embd_scale, 4, bias=configs.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block with gated STU computation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        z = x
        s1 = self.stu_1(x)
        s2 = self.stu_2(x)
        s3 = self.stu_3(x)
        s4 = self.stu_4(x)

        # Stack the outputs
        outputs = torch.stack([s1, s2, s3, s4], dim=-1)

        # Compute the gating weights
        weights = nn.functional.softmax(self.gate(x), dim=-1).unsqueeze(2)

        # Apply the gating weights to the outputs and sum them
        output = (outputs * weights).sum(dim=-1)
        return output + z

class ExponentialLookbackMoE(nn.Module):
    def __init__(self, configs, sigma, V, padded_sl, temperature=1.0, log_buffer_size=100) -> None:
        super(ExponentialLookbackMoE, self).__init__()
        self.stu_1 = STU(configs, sigma, V, padded_sl)
        self.stu_2 = STU(configs, sigma, V, padded_sl)
        self.stu_3 = STU(configs, sigma, V, padded_sl)
        self.stu_4 = STU(configs, sigma, V, padded_sl)

        self.log_weights = nn.Parameter(torch.ones(4))
        self.temperature = temperature
        self.log_file = "stu_usage_data.csv"
        self.step_count = 0
        self.log_buffer = []
        self.log_buffer_size = log_buffer_size

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Timestamp",
                        "Forward Pass",
                        "STU 1 Weight",
                        "STU 2 Weight",
                        "STU 3 Weight",
                        "STU 4 Weight",
                        "Selected STU"
                    ]
                )

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.step_count += 1

        x1, x2, x3, x4 = x, x[:, :x.shape[1]//2, :], x[:, :x.shape[1]//4, :], x[:, :x.shape[1]//8, :]

        s1 = self.stu_1(x1)
        s2 = F.pad(self.stu_2(x2), (0, 0, 0, x1.shape[1] - x2.shape[1]))
        s3 = F.pad(self.stu_3(x3), (0, 0, 0, x1.shape[1] - x3.shape[1]))
        s4 = F.pad(self.stu_4(x4), (0, 0, 0, x1.shape[1] - x4.shape[1]))

        weights = self.gumbel_softmax(self.log_weights, temperature=self.temperature, hard=True)

        outputs = torch.stack([s1, s2, s3, s4], dim=-1)
        output = (outputs * weights.view(1, 1, 1, -1)).sum(dim=-1)

        top_stu_index = torch.argmax(weights).item()

        self.log_current_weights(weights, top_stu_index)
        if len(self.log_buffer) >= self.log_buffer_size:
            self.flush_log_buffer()

        return output

    def log_current_weights(self, weights, selected_stu):
        timestamp = time.time()
        self.log_buffer.append([timestamp, self.step_count] + weights.tolist() + [selected_stu + 1])

    def flush_log_buffer(self):
        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(self.log_buffer)
        self.log_buffer.clear()

    def __del__(self):
        self.flush_log_buffer()  # Ensure any remaining logs are written

class ExponentialLookbackMoE_InputDependent(nn.Module):
    def __init__(self, configs, sigma, V, padded_sl) -> None:
        super(ExponentialLookbackMoE_InputDependent, self).__init__()
        self.stu_1 = STU(configs, sigma, V, padded_sl)
        self.stu_2 = STU(configs, sigma, V, padded_sl)
        self.stu_3 = STU(configs, sigma, V, padded_sl)
        self.stu_4 = STU(configs, sigma, V, padded_sl)
        self.gate_1 = nn.Linear(configs.d_model * configs.embd_scale, 1, bias=configs.bias)
        self.gate_2 = nn.Linear(configs.d_model * configs.embd_scale, 1, bias=configs.bias)
        self.gate_3 = nn.Linear(configs.d_model * configs.embd_scale, 1, bias=configs.bias)
        self.gate_4 = nn.Linear(configs.d_model * configs.embd_scale, 1, bias=configs.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block with gated STU computation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # STU_1 processes the entire input
        x1 = x
        # Apply adaptive regret for other STUs
        x2 = x[:, :x.shape[1]//2, :]  # 1/2 of x
        x3 = x[:, :x.shape[1]//4, :]  # 1/4 of x
        x4 = x[:, :x.shape[1]//8, :]  # 1/8 of x

        # Compute STU outputs
        s1, s2, s3, s4 = self.stu_1(x1), self.stu_2(x2), self.stu_3(x3), self.stu_4(x4)

        # Compute gating weights for each STU
        g1 = self.gate_1(x1).sigmoid()
        g2 = self.gate_2(x2).sigmoid()
        g3 = self.gate_3(x3).sigmoid()
        g4 = self.gate_4(x4).sigmoid()

        # Apply gates to STU outputs
        s1 = s1 * g1
        s2 = s2 * g2
        s3 = s3 * g3
        s4 = s4 * g4

        # Pad STU outputs to match x1's shape
        s2 = F.pad(s2, (0, 0, 0, x1.shape[1] - s2.shape[1]))
        s3 = F.pad(s3, (0, 0, 0, x1.shape[1] - s3.shape[1]))
        s4 = F.pad(s4, (0, 0, 0, x1.shape[1] - s4.shape[1]))

        # Sum the gated and padded outputs
        output = s1 + s2 + s3 + s4

        return output

class SimpleGatedMoe(nn.Module):
    def __init__(self, configs, sigma, V, padded_sl) -> None:
        super(SimpleGatedMoe, self).__init__()
        self.stu_1 = STU(configs, sigma, V, padded_sl)
        self.stu_2 = STU(configs, sigma, V, padded_sl)
        self.stu_3 = STU(configs, sigma, V, padded_sl)
        self.stu_4 = STU(configs, sigma, V, padded_sl)
        self.gate = nn.Linear(configs.d_model * configs.embd_scale, 4, bias=configs.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block with gated STU computation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Compute STU outputs
        s1, s2, s3, s4 = self.stu_1(x), self.stu_2(x), self.stu_3(x), self.stu_4(x)

        # Compute gating weights
        gate_logits = self.gate(x)
        weights = nn.functional.softmax(gate_logits, dim=-1).unsqueeze(2)

        # Stack the outputs
        outputs = torch.stack([s1, s2, s3, s4], dim=-1)

        # Apply the gating weights to the outputs and sum them
        output = (outputs * weights).sum(dim=-1)

        return output

#TODO: instead of MLPs, GELU for non-linearities
class SimplifiedResidualSTU(nn.Module):
    def __init__(self, configs):
        super(SimplifiedResidualSTU, self).__init__()
        self.configs = configs
        self.loss_fn = configs.loss_fn
        self.num_models = configs.num_models
        self.soft_detach_factor = 0.9

        # Input projection
        self.input_proj = nn.Linear(configs.d_in, configs.d_model * configs.embd_scale, bias=configs.bias)

        # Create STU models
        self.models = nn.ModuleList([
            STU(configs, configs.sigma, configs.V, configs.padded_sl)
            for _ in range(self.num_models)
        ])
        
        # Add GELU activation
        self.gelu = nn.GELU()

        # Output projection
        self.output_proj = nn.Linear(configs.d_model * configs.embd_scale, configs.d_out, bias=configs.bias)

        # Report the number of parameters
        print("\nSTU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, list, list]]:
        # Apply input projection
        x = self.input_proj(inputs)

        residual = targets
        all_preds = []
        all_targets = []
        individual_metrics = []

        for i, model in enumerate(self.models):
            preds = model(x)
            preds = self.output_proj(preds)
            all_preds.append(preds)
            all_targets.append(residual)
            
            individual_metrics.append({
                f"model_{i}_pred": preds,
                f"model_{i}_target": residual,
            })
            
            residual = residual - preds.detach()
            
            # Apply GELU activation between models, except for the last one
            if i < len(self.models) - 1:
                residual = self.gelu(residual)

        # Sum all predictions and apply output projection
        final_preds = sum(all_preds)

        return final_preds, (all_preds, all_targets, individual_metrics)
    
    def get_num_params(self):
        """
        Return the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

class ResidualSTU(nn.Module):
    def __init__(self, configs):
        super(ResidualSTU, self).__init__()
        self.configs = configs
        self.loss_fn = configs.loss_fn
        self.num_models = configs.num_models
        self.soft_detach_factor = 0.9
        self.l2_reg_factor = 0.01

        # TODO: Initialize each sub-model with a different random seed
        self.models = nn.ModuleList()
        for i in range(self.num_models):
            seed = random.randint(0, 2**32 - 1)  # Generate a random seed
            torch.manual_seed(seed)
            self.models.append(SpectralSSM(configs))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, list, list]]:
        residual = targets
        all_preds = []
        all_targets = []
        individual_metrics = []

        for i, model in enumerate(self.models):
            preds, _ = model(inputs, residual)
            all_preds.append(preds)
            all_targets.append(residual)
            
            individual_metrics.append({
                f"model_{i}_pred": preds,
                f"model_{i}_target": residual,
            })
            
            residual = residual - preds.detach()

        final_preds = sum(all_preds)

        return final_preds, (all_preds, all_targets, individual_metrics)

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
        self.rn = RMSNorm(configs.embd_scale * configs.d_model)
        self.stu = STU(configs, sigma, V, padded_sl)
        # self.stu = ExponentialLookbackMoE(configs, sigma, V, padded_sl)

        self.mlp = (
            MoE(
                configs,
                experts=[GatedMLP(configs) for _ in range(configs.num_experts)],
                gate=nn.Linear(configs.d_model * configs.embd_scale, configs.num_experts, bias=configs.bias),
            )
            if configs.moe
            else GatedMLP(configs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        z = x
        x = x + self.stu(self.rn(x))
        x = x + self.mlp(x)
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
        self.d_model = configs.d_model
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.embd_scale = configs.embd_scale
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

        self.input_proj = nn.Linear(self.d_in, self.d_model * self.embd_scale, bias=self.bias)

        self.spectral_ssm = nn.ModuleDict(
            dict(
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList(
                    [Block(
                        self.configs, self.sigma, self.V, self.padded_sl
                    ) for _ in range(self.n_layers)]),
            )
        )
        self.output_proj = nn.Linear(self.d_model * self.embd_scale, self.d_out, bias=self.bias)

        # Initialize all weights
        self.m_x = (self.d_model * self.embd_scale)**-0.5
        self.std = (self.d_model * self.embd_scale)**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print("\nSTU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        x = self.spectral_ssm.dropout(x)
        for block in self.spectral_ssm.hidden:
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
                    module.M_y[:, 1] = self.alpha * torch.eye(module.d_model * module.embd_scale)

    def get_num_params(self):
        """
        Return the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        param_dict = {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}
        total_params = sum(param_dict.values())
        
        print("Top 10 parameter groups:")
        for i, (name, count) in enumerate(sorted(param_dict.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"{i}. {name}: {count:,} parameters")
        
        return total_params

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

        return (
            power_flops
            + sum_flops
            + mean_flops
            + sqrt_flops
            + normalize_flops
            + scale_flops
        )

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
        rollout_steps: int = 1,
        truth: int = 0
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
            rollout_steps (int): Number of predicted steps to calculate the mean loss over
            truth (int): Interval at which to ground predictions to true targets.
                If 0, no grounding is performed.

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
        _, _, d_in = inputs.size()
        assert init + steps <= total_steps, f"Cannot take more steps than {total_steps}"
        assert rollout_steps <= steps, "Cannot roll out for more than total steps"

        # Track model hallucinations
        predicted_steps = torch.zeros(num_traj, steps, d_out, device=device)

        # Track loss between rollout vs ground truth
        traj_losses = torch.zeros(num_traj, steps, device=device)

        # Initialize cost function
        mse_loss = nn.MSELoss()

        # Initialize autoregressive inputs with all available context
        ar_inputs = inputs[:, :init].clone()

        for step in tqdm(range(steps), desc="Predicting", unit="step"):
            current_step = init + step

            # Predict the next state using a fixed window size of inputs
            step_preds, (_, _) = self.forward(
                ar_inputs[:, :current_step], targets[:, :current_step]
            )

            # Calculate the mean loss of the last rollout_steps predictions
            rollout_preds = step_preds[:, -rollout_steps:, :]
            rollout_ground_truths = targets[:, (current_step - rollout_steps) : current_step, :]
            traj_losses[:, step] = mse_loss(rollout_preds, rollout_ground_truths)

            # Store the last prediction step for plotting
            predicted_steps[:, step] = step_preds[:, -1].squeeze(1)

            # Decide whether to use the prediction or ground truth as the next input
            if truth == 0 or (step + 1) % truth == 0:
                # Concatenate the autoregressive predictions of states and the ground truth actions
                next_input = step_preds[:, -1:].detach()
                next_action = inputs[:, current_step:current_step+1, -(d_in - d_out):]
                next_input = torch.cat([next_input, next_action], dim=2)
                ar_inputs = torch.cat([ar_inputs, next_input], dim=1)
            else:
                next_input = inputs[:, current_step:current_step+1, :]
                ar_inputs = torch.cat([ar_inputs, next_input], dim=1)

        avg_loss = traj_losses.mean()
        return predicted_steps, (avg_loss, traj_losses)
