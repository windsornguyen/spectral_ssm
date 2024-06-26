# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Isabel Liu, Yagiz Devre
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import math

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from models.stu.stu_utils import (
    compute_ar_x_preds,
    compute_x_tilde,
    compute_y_t,
    get_top_hankel_eigh,
)
from time import time
from utils.swiglu import SwiGLU
from utils.rms_norm import RMSNorm
from tqdm import tqdm


@dataclass
class SSSMConfigs:
    d_in: int = 29
    d_out: int = 29
    n_layers: int = 4
    n_embd: int = 512
    sl: int = 300
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 24
    k_u: int = 3  # Number of parametrizable, autoregressive matrices Mᵘ
    k_y: int = 2  # Number of parametrizable, autoregressive matrices Mʸ
    learnable_m_y: bool = True
    alpha: float = (
        0.9  # 0.9 deemed "uniformly optimal" in the paper # TODO: Add this in train.py
    )
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )


class STU(nn.Module):
    """
    A simple STU (Spectral Transform Unit) Layer.

    Args:
        d_out (int): Output dimension.
        sl (int): Input sequence length.
        num_eigh (int): Number of eigenvalues and eigenvectors to use.
        k_u (int): Auto-regressive depth on the input sequence.
        k_y (int): Auto-regressive depth on the output sequence.
        learnable_m_y (bool): Whether the m_y matrix is learnable.
    """

    def __init__(self, configs) -> None:
        super(STU, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_out = configs.d_out
        self.l, self.k = configs.sl, configs.num_eigh
        self.eigh = get_top_hankel_eigh(self.l, self.k, self.device)
        self.k_u = configs.k_u
        self.k_y = configs.k_y
        self.learnable_m_y = configs.learnable_m_y
        self.alpha = configs.alpha
        self.m_x = 1.0 / (float(self.d_out) ** 0.5)
        self.m_u = nn.Parameter(torch.empty([self.d_out, self.d_out, self.k_u]))
        self.m_phi = nn.Parameter(torch.empty([self.d_out * self.k, self.d_out]))
        self.m_y = (
            nn.Parameter(torch.empty([self.d_out, self.k_y, self.d_out]))
            if self.learnable_m_y
            else self.register_buffer(
                "m_y", torch.empty([self.d_out, self.k_y, self.d_out])
            )
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, inputs):
        x_tilde = compute_x_tilde(inputs, self.eigh)
        delta_phi = x_tilde @ self.m_phi
        delta_ar_u = compute_ar_x_preds(self.m_u, inputs)
        y_t = compute_y_t(self.m_y, delta_phi + delta_ar_u)
        return self.dropout(y_t)


class MLP(nn.Module):
    """
    Simple feed-forward network.
    """

    def __init__(self, configs):
        super(MLP, self).__init__()
        self.h_dim = (configs.scale * configs.n_embd * 2) // 3
        self.swiglu = SwiGLU(dim=configs.n_embd, h_dim=self.h_dim, bias=configs.bias)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = self.swiglu(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, configs):
        super(Block, self).__init__()
        self.rn_1 = RMSNorm(configs.n_embd)
        self.stu = STU(configs)
        self.rn_2 = RMSNorm(configs.n_embd)
        self.mlp = MLP(configs)

    def forward(self, x):
        z = self.rn_1(x)
        x = self.stu(self.rn_2(x))
        x = x + self.mlp(x)
        return x + z


class SSSM(nn.Module):
    """
    General model architecture based on STU blocks.
    """

    def __init__(self, configs):
        super(SSSM, self).__init__()
        self.configs = configs
        self.n_layers = configs.n_layers
        self.n_embd = configs.n_embd
        self.d_in = configs.d_in
        self.d_out = configs.d_out
        self.sl, self.k = configs.sl, configs.num_eigh

        self.bias = configs.bias
        self.dropout = configs.dropout
        self.loss_fn = configs.loss_fn
        self.controls = configs.controls

        self.emb = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.stu = nn.ModuleDict(
            dict(
                # Since our tasks are continuous, we do not use token embeddings.
                wpe=nn.Embedding(self.sl, self.n_embd),
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList([Block(configs) for _ in range(self.n_layers)]),
            )
        )
        self.task_head = nn.Linear(self.n_embd, self.d_out, bias=self.bias)

        if self.controls["task"] == "mujoco-v1":
            if self.controls["controller"] == "Ant-v1":
                self.d_out = 29
            else:
                self.d_out = 18

        # Initialize weights
        self.m_x = self.d_out**-0.5
        self.std = self.n_embd**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print("STU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias, mean=0.0, std=self.std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            # Custom initialization for m_u, m_phi, and m_y matrices
            torch.nn.init.uniform_(module.m_u, -self.m_x, self.m_x)
            torch.nn.init.xavier_normal_(module.m_phi)
            if module.learnable_m_y:
                torch.nn.init.xavier_normal_(module.m_y)

            # Initialize Mʸ₂ = α * I, page 8.
            # if module.k_y > 1:
            #     with torch.no_grad():
            #         module.m_y[:, 1] = module.alpha * torch.eye(self.d_out)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding (bool, optional):
            Whether to exclude the positional embeddings (if applicable).
            Defaults to True.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def forward(self, inputs, targets):
        bsz, sl, n_embd = inputs.size()

        # Pass inputs through the embedding layer
        x = self.emb(inputs)
        x /= math.log(self.sl)  # <-- From Evan
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

    def predict_states(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int,
        steps: int,
        truth: int = 0,
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
            truth (int): Interval at which to ground predictions to true targets.
                         If 0, no grounding is performed.

        Returns:
        tuple: Contains the following elements:
            - preds (torch.Tensor): Predictions of shape (num_traj, total_steps, d_out)
            - tuple:
                - avg_loss (torch.Tensor): Scalar tensor with the average loss
                - avg_metrics (dict[str, torch.Tensor]): Dictionary of average metrics, each a scalar tensor
                - traj_losses (torch.Tensor): Losses for each trajectory and step, shape (num_traj, steps)
                - metrics (dict[str, torch.Tensor]): Detailed metrics for each trajectory and step, each of shape (num_traj, steps)
        """
        device = next(self.parameters()).device
        print(f"Predicting on {device}.")
        num_traj, total_steps, d_in = inputs.size()
        _, _, d_out = targets.size()
        
        assert (init + steps <= total_steps), f"init ({init}) + steps ({steps}) must be <= total_steps ({total_steps})"
        assert truth >= 0, "The 'truth' parameter must be non-negative."
        assert total_steps % truth == 0, "The total number of steps must be divisible by the 'truth' parameter."

        # Optimization for truth == 1 case
        if truth == 1:
            preds = targets.clone()
            traj_losses = torch.zeros(num_traj, steps, device=device)
            metrics = {
                key: torch.zeros(num_traj, steps, device=device)
                for key in [
                    "coordinate_loss",
                    "orientation_loss",
                    "angle_loss",
                    "coordinate_velocity_loss",
                    "angular_velocity_loss",
                ]
            }
            for step in range(steps):
                _, (step_loss, step_metrics) = self.forward(
                    inputs[:, :init+step], targets[:, :init+step]
                )
                traj_losses[:, step] = step_loss.squeeze()
                for key in metrics:
                    metrics[key][:, step] = step_metrics[key]
            
            avg_loss = traj_losses.mean()
            avg_metrics = {key: metrics[key].mean() for key in metrics}
            return preds, (avg_loss, avg_metrics, traj_losses, metrics)

        metrics = {
            key: torch.zeros(num_traj, steps, device=device)
            for key in [
                "coordinate_loss",
                "orientation_loss",
                "angle_loss",
                "coordinate_velocity_loss",
                "angular_velocity_loss",
            ]
        }

        # Initialize predictions tensor
        preds = torch.zeros(num_traj, total_steps, d_out, device=device)
        traj_losses = torch.zeros(num_traj, steps, device=device)

        # Copy over ground truth values up to init for context
        preds[:, :init] = targets[:, :init]

        # Initialize autoregressive inputs with all available context
        ar_inputs = inputs[:, :init].clone()

        for step in tqdm(range(steps), desc="Predicting", unit="step"):
            current_step = init + step

            # Predict the next state using all available autoregressive inputs
            step_preds, (step_loss, step_metrics) = self.forward(
                ar_inputs, targets[:, :current_step]
            )

            # Decide whether to use the prediction or ground truth as the next input
            if truth > 0 and (step + 1) % truth == 0:
                next_input = inputs[:, current_step].unsqueeze(1)
                preds[:, current_step] = targets[:, current_step]
            else:
                next_input = step_preds[:, -1:].detach()
                preds[:, current_step] = step_preds[:, -1].squeeze(1)

            # Append the next input to ar_inputs, maintaining full history
            ar_inputs = torch.cat([ar_inputs, next_input], dim=1)

            # Track the loss for current prediction
            traj_losses[:, step] = step_loss.squeeze()

            # Track the metrics for current prediction
            for key in metrics:
                metrics[key][:, step] = step_metrics[key]

        avg_loss = traj_losses.mean()
        avg_metrics = {key: metrics[key].mean() for key in metrics}

        return preds, (avg_loss, avg_metrics, traj_losses, metrics)