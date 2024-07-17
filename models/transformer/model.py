# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: (Transformer) model.py
# =============================================================================#

import math
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange
from torch.nn import functional as F
from tqdm import tqdm
from models.transformer.attn import CausalSelfAttention
from models.transformer.dilated.dilated_attn import DilatedCausalSelfAttention
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU
from utils.dist_utils import all_gather_func, get_data_parallel_rank, get_data_parallel_world_size


@dataclass
class TransformerConfigs:
    n_layers: int = 2
    n_embd: int = 512  # Embedding dimension
    n_heads: int = 16  # Constraint: n_heads % n_embd == 0
    sl: int = 300  # Sequence length
    scale: int = 4
    sub_rn: bool = True
    bias: bool = False
    dropout: float = 0.10
    flash_attn: bool = True
    use_sq_relu: bool = False
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )
    device: torch.device = None
    
    # MoE
    moe: bool = True
    num_experts: int = 8
    num_experts_per_timestep: int = 2

    # Dilated Attention
    dilated_attn: bool = False
    segment_lengths: list[int] = field(default_factory=lambda: [128]) # TODO: Check this makes sense (and follows paper)
    dilated_ratios: list[int] = field(default_factory=lambda: [1]) # TODO: Check this makes sense (and follows paper)
    seq_parallel: bool = True
    xpos_rel_pos: bool = True
    xpos_scale_base: int = 512
    rms_norm_eps: float = 1e-5
    multiway: bool = False

class FFN(nn.Module):
    """
    Feed-forward network using the SwiGLU activation.

    Args:
        configs: Configuration object containing the following attributes:
            scale (float): Scaling factor for hidden dimension.
            n_embd (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super(FFN, self).__init__()
        self.swiglu = SwiGLU(
            dim=configs.n_embd, h_dim=configs.scale * configs.n_embd,
            bias=configs.bias, use_sq_relu=configs.use_sq_relu
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
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

class GatedFFN(nn.Module):
    """
    Gated feed-forward network using SiLU activation.

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
        Forward pass of the GatedFFN.

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

class TransformerBlock(nn.Module):
    """
    Single block of the Transformer.
    """

    def __init__(self, configs):
        super(TransformerBlock, self).__init__()
        self.configs = configs
        self.attn = self._get_attn_type(configs)
        self.rn_1 = RMSNorm(configs.n_embd, eps=configs.rms_norm_eps)
        self.rn_2 = RMSNorm(configs.n_embd, eps=configs.rms_norm_eps)

        self.ffn_1 = MoE(
            configs,
            experts=[GatedFFN(configs) for _ in range(configs.num_experts)],
            gate=nn.Linear(configs.n_embd, configs.num_experts, bias=configs.bias)
        ) if configs.moe else GatedFFN(configs)

    def _get_attn_type(self, configs):
        if configs.dilated_attn:
            return DilatedCausalSelfAttention(configs)
        else:
            return CausalSelfAttention(configs)

    def forward(self, x):
        x = self.rn_1(x)
        x = x + self.attn(x)
        x = x + self.ffn_1(self.rn_2(x))
        return x

class Transformer(nn.Module):
    """
    Transformer architecture adapted from the GPT-2 implementation.
    """

    def __init__(self, configs):
        super(Transformer, self).__init__()
        assert configs.sl is not None
        self.configs = configs
        self.controls = configs.controls
        self.n_embd = configs.n_embd
        self.d_in = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(self.configs.dropout)
        
        if configs.moe:
            print(f"\nMoE?: Enabled | Using {configs.num_experts} experts.")
        else:
            print(f"\nMoE?: Disabled")

        self.transformer = nn.ModuleDict(
            dict(
                # Since our tasks are continuous, we do not use token embeddings.
                wpe=nn.Embedding(configs.sl, configs.n_embd),
                dropout=self.dropout,
                hidden=nn.ModuleList(
                    [TransformerBlock(configs) for _ in range(configs.n_layers)]
                ),
            )
        )

        # Adjust output dims based on task and controller
        self.d_out = configs.n_embd
        if configs.controls["task"] == "mujoco-v1":
            if configs.controls["controller"] == "Ant-v1":
                self.d_out = 29
            else:
                self.d_out = 18

        self.output = nn.Linear(configs.n_embd, self.d_out, bias=configs.bias)
        self.loss_fn = self.configs.loss_fn

        # Initialize all weights
        self.std = self.n_embd**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print(
            "\nTransformer Model Parameter Count (excl. pos. emb.): %.2fM"
            % (self.get_num_params() / 1e6,)
        )

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                # Scale by 2 to account for self-attn and ffn sub-layer
                self.std *= (2 * self.configs.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Returns:
            int: The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, inputs, targets=None):
        """
        Perform the forward pass of the Transformer model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
            targets (torch.Tensor, optional): Target tensor for training

        Returns:
            torch.Tensor: Predicted output tensor of shape (bsz, sl, d_out)
            tuple: Loss (and metrics, if applicable)
        """
        bsz, sl, d_in = inputs.size()

        # Generate positional embeddings for the sequence
        pos = torch.arange(0, sl, dtype=torch.long, device=inputs.device)  # -> (sl)

        # Position embeddings of shape (sl, n_embd)
        pos_emb = self.transformer.wpe(pos)  # -> (sl, n_embd)

        # Add positional embeddings to the input
        x = inputs + pos_emb

        incremental_state = None # 
        if self.configs.dilated_attn:
            incremental_state = {}

        x = self.transformer.dropout(inputs)

        # Pass through each transformer block in hidden layers
        for block in self.transformer.hidden:
            x = block(x)

        # Output model predictions!
        preds = self.output(x)  # -> (bsz, sl, d_out)

        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            return preds, loss

    # TODO: Not sure when/where this could be used, but we'd like to use it!
    # TODO: Also need to fix this function to make sure it's correct.
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.configs
        L, H, Q, T = (
            cfg.num_layers,
            cfg.n_heads,
            cfg.d_embd // cfg.n_heads,
            cfg.ctxt_len,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def flops_per_token(self):
        """Estimate the number of floating-point operations per token."""
        flops = 0
        cfg = self.configs
        # Embedding layers
        flops += 2 * cfg.d_model * cfg.max_seq_len  # input and position embeddings
        # Transformer blocks
        for _ in range(cfg.num_layers):
            # Layer normalization
            flops += 4 * cfg.d_model * cfg.max_seq_len  # ln_1 and ln_2
            # Multi-head attention
            flops += (
                2 * cfg.num_heads * cfg.d_model * cfg.max_seq_len
            )  # Compute query, key, value
            flops += (
                2 * cfg.d_model * cfg.num_heads * cfg.d_model
            )  # Apply attention weights
            # FFN layer
            flops += 2 * cfg.d_model * cfg.d_ff * cfg.d_model  # fc_1
            flops += cfg.d_ff * cfg.d_model  # Activation function
            flops += 2 * cfg.d_ff * cfg.d_model * cfg.d_model  # fc_2
        # Final layer normalization
        flops += 4 * cfg.d_model * cfg.max_seq_len  # ln_f
        # Language model head
        flops += 2 * cfg.d_model * cfg.vocab_size
        return flops

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
        mse_loss = nn.MSELoss()

        for step in tqdm(range(steps), desc="Predicting", unit="step"):
            current_step = init + step

            # Predict the next state using a fixed window size of inputs
            step_preds, (_, _) = self.forward(
                inputs[:, :current_step], targets[:, :current_step]
            )

            # Calculate the mean loss of the last rollout_steps predictions
            rollout_preds = step_preds[:, -rollout_steps:, :]
            rollout_ground_truths = targets[:, (current_step - rollout_preds.shape[1]) : current_step, :]
            traj_losses[:, step] = mse_loss(rollout_preds, rollout_ground_truths)

            # Store the last prediction step for plotting
            predicted_steps[:, step] = step_preds[:, -1].squeeze(1)

            # # Concatenate the autoregressive predictions of states and the ground truth actions (hallucinating)
            # next_action = inputs[:, current_step:current_step+1, -(d_in - d_out):]
            # next_input = torch.cat([next_input, next_action], dim=2)
            # ar_inputs = torch.cat([ar_inputs, next_input], dim=1)

        avg_loss = traj_losses.mean()
        return predicted_steps, (avg_loss, traj_losses)
