# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from models.stu.stu_utils import get_top_eigh, compute_ar_u, compute_spectral, compute_ar_y
from utils.swiglu import SwiGLU
from utils.rms_norm import RMSNorm
from tqdm import tqdm
from torch.nn import MSELoss


@dataclass
class SSSMConfigs:
    d_in: int = 37
    d_out: int = 37
    d_proj: int = 29
    n_layers: int = 6
    n_embd: int = 37
    sl: int = 300
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 24
    k_u: int = 3  # Number of parametrizable, autoregressive matrices Mᵘ
    k_y: int = 2  # Number of parametrizable, autoregressive matrices Mʸ
    learnable_m_y: bool = True
    alpha: float = 0.9  # 0.9 deemed "uniformly optimal" in the paper
    use_hankel_L: bool = False
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )


class STU(nn.Module):
    """
    A simple STU (Spectral Transform Unit) layer.

    Args:
        configs: Configuration object containing the following attributes:
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            sl (int): Input sequence length.
            num_eigh (int): Number of spectral filters to use.
            k_u (int): Autoregressive depth on the input sequence.
            k_y (int): Autoregressive depth on the output sequence.
            use_hankel_L (bool): Use the alternative Hankel matrix?
            learnable_m_y (bool): Learn the M_y matrix?
            dropout (float): Dropout rate.
    """

    def __init__(self, configs) -> None:
        super(STU, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.l, self.k = configs.sl, configs.num_eigh
        self.use_hankel_L = configs.use_hankel_L
        self.eigh = get_top_eigh(self.l, self.k, self.use_hankel_L, self.device)
        self.k_u = configs.k_u
        self.k_y = configs.k_y
        self.learnable_m_y = configs.learnable_m_y
        self.dropout = nn.Dropout(configs.dropout)

        # Parameterizable matrix Mᵘ, Mᵠ⁺, and Mᵠ⁻, per section 3
        self.M_u = nn.Parameter(torch.empty(self.k_u, self.d_out, self.d_in))
        self.M_phi_plus = nn.Parameter(torch.empty(self.k, self.d_out, self.d_in))
        self.M_phi_minus = nn.Parameter(torch.empty(self.k, self.d_out, self.d_in))

        # Parametrizable matrix Mʸ Introduced in section 5, equation 5
        if self.learnable_m_y:
            # TODO: Change this to (d_out, k_y, d_out) so that shaping in compute_ar_y is easier
            self.M_y = nn.Parameter(torch.zeros(self.k_y, self.d_out, self.d_out))
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
        ar_u = compute_ar_u(self.M_u, inputs)
        M_y = self.M_y if self.learnable_m_y else None
        spectral = compute_spectral(inputs, self.eigh, self.M_phi_plus, self.M_phi_minus, M_y)

        y_t = ar_u + spectral
        if self.learnable_m_y:
            y_t += compute_ar_y(self.M_y, y_t)

        return self.dropout(y_t)


class MLP(nn.Module):
    """
    Simple multi-layer perceptron network using SwiGLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            scale (float): Scaling factor for hidden dimension.
            n_embd (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs: SSSMConfigs) -> None:
        super(MLP, self).__init__()
        self.h_dim = configs.scale * configs.n_embd
        self.swiglu = SwiGLU(dim=configs.n_embd, h_dim=self.h_dim, bias=configs.bias)
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
        return self.dropout(x)


class Block(nn.Module):
    """
    A single block of the SSSM model, consisting of STU and MLP layers.

    Args:
        configs: Configuration object for STU and MLP layers
    """

    def __init__(self, configs: SSSMConfigs) -> None:
        super(Block, self).__init__()
        self.rn_1 = RMSNorm(configs.n_embd)
        self.stu = STU(configs)
        self.rn_2 = RMSNorm(configs.n_embd)
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
        x = self.stu(self.rn_2(x))
        x = x + self.mlp(x)
        return x + z


class SSSM(nn.Module):
    """
    General model architecture based on stacked STU blocks and MLP layers.

    Args:
        configs: Configuration object containing model parameters
    """

    def __init__(self, configs: SSSMConfigs) -> None:
        super(SSSM, self).__init__()
        self.configs = configs
        self.n_layers = configs.n_layers
        self.n_embd = configs.n_embd
        self.d_in = configs.d_in    # TODO: d_in is not exactly the same as n_embd
        self.d_out = configs.d_out
        self.d_proj = configs.d_proj
        self.sl = configs.sl
        self.learnable_m_y = configs.learnable_m_y
        self.alpha = configs.alpha

        self.bias = configs.bias
        self.dropout = configs.dropout
        self.loss_fn = configs.loss_fn
        self.controls = configs.controls

        self.emb = nn.Linear(self.n_embd, self.n_embd)
        self.stu = nn.ModuleDict(
            dict(
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList([Block(configs) for _ in range(self.n_layers)]),
            )
        )
        self.task_head = nn.Linear(self.n_embd, self.d_proj, bias=self.bias)

        # Initialize all weights
        self.m_x = self.d_out**-0.5
        self.std = self.n_embd**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print("STU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, inputs, targets):
        """
        Forward pass of the SSSM model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor): Target tensor for loss computation

        Returns:
            Type (ignore due to high variability):
            - Predictions tensor
            - Tuple containing loss and metrics (if applicable)
        """
        _, sl, n_embd = inputs.size()

        x = self.emb(inputs)
        x = self.stu.dropout(x)

        for block in self.stu.hidden:
            x = block(x)
        
        preds = self.task_head(x)

        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            return preds, (loss,)

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
                    module.M_y[1, :, :] = self.alpha * torch.eye(module.d_out)

    def get_num_params(self):
        """
        Return the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    # TODO: Not sure when/where this could be used, but we'd like to use it!
    # TODO: Also need to fix this function to make sure it's correct.
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.configs
        L, D, E, T = cfg.num_layers, cfg.n_embd, cfg.num_eigh, cfg.input_len

        # Embedding layers
        embed_flops = 2 * D * T

        # STU blocks
        stu_block_flops = 0
        for _ in range(L):
            # Layer normalization
            stu_block_flops += 2 * D * T  # ln_1 and ln_2

            # STU layer
            stu_block_flops += 2 * E * D * T  # Compute x_tilde
            stu_block_flops += 2 * D * E * D  # Apply m_phi matrix

            # MLP layer
            stu_block_flops += 2 * D * cfg.scale * D  # c_fc
            stu_block_flops += cfg.scale * D  # GELU activation
            stu_block_flops += 2 * cfg.scale * D * D  # c_proj

        # Final layer normalization
        final_ln_flops = 2 * D * T  # ln_f

        # Language model head
        lm_head_flops = 2 * D * cfg.vocab_size

        flops_per_iter = embed_flops + stu_block_flops + final_ln_flops + lm_head_flops
        flops_per_fwdbwd = flops_per_iter * fwdbwd_per_iter

        # Express flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_fwdbwd / dt  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # TODO: Also need to fix this function to make sure it's correct.
    def flops_per_token(self):
        """Estimate the number of floating-point operations per token."""
        flops = 0
        cfg = self.configs

        # Embedding layers
        flops += 2 * cfg.n_embd * cfg.block_size  # wte and wpe embeddings

        # STU blocks
        for _ in range(cfg.num_layers):
            # Layer normalization
            flops += 2 * cfg.n_embd * cfg.block_size  # ln_1 and ln_2

            # STU layer
            flops += 2 * cfg.num_eigh * cfg.n_embd * cfg.block_size  # Compute x_tilde
            flops += 2 * cfg.n_embd * cfg.num_eigh * cfg.n_embd  # Apply m_phi matrix

            # MLP layer
            flops += 2 * cfg.n_embd * cfg.scale * cfg.n_embd  # c_fc
            flops += cfg.scale * cfg.n_embd  # GELU activation
            flops += 2 * cfg.scale * cfg.n_embd * cfg.n_embd  # c_proj

        # Final layer normalization
        flops += 2 * cfg.n_embd * cfg.block_size  # ln_f

        # Language model head
        flops += 2 * cfg.n_embd * cfg.vocab_size

        return flops

    def predict_states(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 950,
        steps: int = 50,
        rollout_steps: int = 20,
        window_size: int = 900,
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
        num_traj, _, d_out = targets.size()

        # To track what the model hallucinates
        hallucinated_steps = torch.zeros(num_traj, steps, d_out, device=device)

        # To track the MSE loss between rollout vs ground truth for each trajectory
        traj_losses = torch.zeros(num_traj, steps, device=device)

        for step in tqdm(range(steps), desc="Predicting", unit="step"):
            current_step = init + step # Start at init
            window_start = current_step - window_size

            # Predict the next state using a fixed window size of inputs
            step_preds, (_, _) = self.forward(
                inputs[:, window_start:current_step], targets[:, window_start:current_step]
            )

            # Calculate the mean loss of the last rollout_steps predictions
            rollout_preds = step_preds[:, -rollout_steps:, :]
            rollout_ground_truths = targets[:, (current_step + 1 - rollout_steps) : (current_step + 1), :]
            mse_loss = MSELoss()
            traj_losses[:, step] = mse_loss(rollout_preds, rollout_ground_truths)

            # Store the last prediction step for plotting
            hallucinated_steps[:, step] = step_preds[:, -1].squeeze(1)

            # # Concatenate the autoregressive predictions of states and the ground truth actions
            # next_action = inputs[:, current_step:current_step+1, -(d_in - d_out):]
            # next_input = torch.cat([next_input, next_action], dim=2)
            # ar_inputs = torch.cat([ar_inputs, next_input], dim=1)

        avg_loss = traj_losses.mean()
        return hallucinated_steps, (avg_loss, traj_losses)
