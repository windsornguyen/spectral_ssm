# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Isabel Liu
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

import math

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from models.stu.stu_utils import compute_ar, compute_spectral, get_top_eigh
from utils.swiglu import SwiGLU
from utils.rms_norm import RMSNorm
from tqdm import tqdm


@dataclass
class SSSMConfigs:
    d_in: int = 24
    d_out: int = 18
    n_layers: int = 6
    n_embd: int = 512
    sl: int = 300
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 32
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
        self.m_u = nn.Parameter(torch.empty(self.k_u, self.d_out, self.d_in))
        self.m_phi_plus = nn.Parameter(torch.empty(self.k, self.d_out, self.d_in))
        self.m_phi_minus = nn.Parameter(torch.empty(self.k, self.d_out, self.d_in))

        # Parametrizable matrix Mʸ Introduced in section 5, equation 5
        if self.learnable_m_y:
            self.m_y = nn.Parameter(torch.zeros(self.k_y, self.d_out, self.d_out))
        else:
            self.register_buffer("m_y", torch.zeros(self.k_y, self.d_out, self.d_out))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STU layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (bsz, sl, d_out)
        """
        bsz, sl, _ = inputs.shape
        # FIXME: shift bug 2
        preds = torch.zeros_like(inputs) # TODO: We don't take outputs into account...
        # TODO: Would this preds state be in training loop and fed back into model?
        # TODO: That doesn't make sense since each batch is independent...
        # autoregressive = compute_ar(self.m_y, preds, self.m_u, inputs)
        spectral = compute_spectral(inputs, self.eigh, self.m_phi_plus, self.m_phi_minus, self.k_y)

        # y_t = autoregressive + spectral # FIXME: shift bug 3
        y_t = spectral
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
        x = x + self.stu(self.rn_1(x))
        x = x + self.mlp(self.rn_2(x))
        return x


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
        self.d_out = configs.d_out
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
        self.task_head = nn.Linear(self.n_embd, self.d_out, bias=self.bias)

        if self.controls["task"] == "mujoco-v1":
            if self.controls["controller"] == "Ant-v1":
                self.d_out = 29
            else:
                self.d_out = 18

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
        x /= math.log(self.sl)  # <-- Suggestion from Evan
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
            torch.nn.init.uniform_(module.m_u, -self.m_x, self.m_x)
            torch.nn.init.xavier_normal_(module.m_phi_plus)
            torch.nn.init.xavier_normal_(module.m_phi_minus)

            # Initialize Mʸ₂ = α * I, page 8.
            if self.learnable_m_y and module.k_y > 1:
                with torch.no_grad():
                    module.m_y[1] = self.alpha * torch.eye(module.d_out)

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
        init: int = 0,
        steps: int = 100,
        ar_steps: int = 1000,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Predicts the next states for a given set of input trajectories using vectorized operations.

        Args:
            inputs (torch.Tensor): A tensor of input trajectories with shape [num_trajectories, seq_len, d_in].
            targets (torch.Tensor): A tensor of target trajectories with shape [num_trajectories, seq_len, d_out].
            init (int): The index of the initial state to start the prediction from. Defaults to 0.
            steps (int): The number of time steps to predict. Defaults to 100.
            ar_steps (int): The number of autoregressive steps to take before using the ground truth state.
                Defaults to 1, which means the model always uses the ground truth state to predict the next state.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
                - preds (torch.Tensor): A tensor of predicted states for each trajectory after `steps` time steps,
                    with shape [num_trajectories, steps, d_out].
                - loss (tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]): A tuple containing:
                    - avg_loss (torch.Tensor): The mean loss over time steps and trajectories.
                    - avg_metrics (dict[str, torch.Tensor]): A dictionary of mean losses for each metric.
                    - trajectory_losses (torch.Tensor): A tensor of losses for each trajectory at each time step,
                        with shape [num_trajectories, steps].
        """

        device = next(self.parameters()).device
        print(f"Predicting on {device}.")
        num_trajectories, sl, d_in = inputs.size()

        # Initialize the predicted sequences and losses
        ar_sequences = inputs.clone()
        preds = torch.zeros(num_trajectories, steps, self.configs.d_out, device=device)
        trajectory_losses = torch.zeros(num_trajectories, steps, device=device)
        metrics = {
            key: torch.zeros(num_trajectories, steps, device=device)
            for key in [
                "coordinate_loss",
                "orientation_loss",
                "angle_loss",
                "coordinate_velocity_loss",
                "angular_velocity_loss",
            ]
        }

        i = init
        with tqdm(total=steps, desc="Predicting", unit="step") as pbar:
            while i < init + steps:
                window_start = max(0, i - self.configs.sl + 1)

                input_window = ar_sequences[:, window_start : i + 1, :]
                target_window = targets[:, window_start : i + 1, :]
                preds_step, (step_loss, step_metrics) = self.forward(
                    input_window, target_window
                )

                preds[:, i - init, :] = preds_step[:, -1, :]
                trajectory_losses[:, i - init] = step_loss

                for key in metrics:
                    metrics[key][:, i] = step_metrics[key]

                # Update autoregressive sequences for the next step
                if i < init + steps - 1:
                    next_step = i + 1
                    if next_step < sl:
                        next_input = (
                            preds[:, i - init, :]
                            if (i - init + 1) % ar_steps != 0
                            else inputs[:, next_step, :]
                        )
                        ar_sequences[:, next_step, :] = next_input
                    else:
                        ar_sequences = torch.cat(
                            [
                                ar_sequences[:, 1:, :],
                                preds[:, i - init : i - init + 1, :],
                            ],
                            dim=1,
                        )

                i += 1
                pbar.update(1)

        avg_loss = trajectory_losses.mean()
        avg_metrics = {key: metrics[key].mean() for key in metrics}

        return preds, (avg_loss, avg_metrics, trajectory_losses, metrics)

    def predict_frames(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 140,
        steps: int = 5,
        ar_steps: int = 300,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Predicts the video frame.

        Args:
            inputs (torch.Tensor): A tensor of input videos with shape [num_videos, sl, d_in].
            targets (torch.Tensor): A tensor of target videos with shape [num_videos, sl, d_in].
            init (int): The index of the initial state to start the prediction from. Defaults to 0.
            steps (int): The number of time steps to predict. Defaults to 50.
            ar_steps (int): The number of autoregressive steps to take before using the ground truth state.
                Defaults to 1, which means the model always uses the ground truth state to predict the next state.
                If set to sl, the model always uses the last predicted state to predict the next state.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
                - preds (torch.Tensor): A tensor of predicted states for each video after `steps` time steps,
                    with shape [num_videos, steps, d_out].
                - loss (tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]): A tuple containing:
                    - avg_loss (torch.Tensor): The mean loss over time steps and videos.
                    - video_losses (torch.Tensor): A tensor of losses for each video at each time step,
                        with shape [num_videos, steps].
        """
        device = next(self.parameters()).device
        print(f"Predicting on {device}.")
        num_videos, sl, d_in = inputs.size()

        # Initialize the predicted sequences and losses
        ar_sequences = inputs.clone()
        preds = torch.zeros(num_videos, steps, d_in, device=device)
        video_losses = torch.zeros(num_videos, steps, device=device)

        i = init
        with tqdm(total=steps, desc="Predicting", unit="step") as pbar:
            while i < init + steps:
                window_start = max(0, i - self.configs.sl + 1)  # TODO: fixed window?

                input_window = ar_sequences[:, window_start : i + 1, :]
                target_window = targets[:, window_start : i + 1, :]
                preds_step, (step_loss,) = self.forward(input_window, target_window)

                preds[:, i - init, :] = preds_step[:, -1, :]
                video_losses[:, i - init] = step_loss

                # Update autoregressive sequences for the next step
                if i < init + steps - 1:
                    next_step = i + 1
                    if next_step < sl:
                        next_input = (
                            preds[:, i - init, :]
                            if (i - init + 1) % ar_steps != 0
                            else inputs[:, next_step, :]
                        )
                        ar_sequences[:, next_step, :] = next_input
                    else:
                        ar_sequences = torch.cat(
                            [
                                ar_sequences[:, 1:, :],
                                preds[:, i - init : i - init + 1, :],
                            ],
                            dim=1,
                        )

                i += 1
                pbar.update(1)

        # # If we've reached the end of the input sequence but still have steps to predict,
        # # use the last predicted state as input (we need to hallucinate and autoregressively predict)
        # for step in range(sl - init, steps):
        #     xs = ar_sequences[:, -1, :].unsqueeze(1)
        #     ys = None

        #     preds_step, step_loss = self.forward(xs, ys)

        #     preds[:, i, :] = preds_step[:, -1, :]

        #     # Update autoregressive sequences for each video independently
        #     if step < steps - 1:
        #         for video_idx in range(num_videos):
        #             next_input = ar_sequences[video_idx, -1, :].clone()
        #             next_input = preds[video_idx, i, :]
        #             ar_sequences[video_idx] = ar_sequences[video_idx, step + 1 + init, :] = next_input

        #     video_losses[:, i] = step_loss

        # Calculate average losses and metrics across videos
        avg_loss = video_losses.mean()

        return preds, (avg_loss, video_losses)
