# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: (Transformer) model.py
# =============================================================================#

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from tqdm import tqdm
from models.transformer.attn import CausalSelfAttention
from models.transformer.dilated.dilated_attn import DilatedCausalSelfAttention
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU


@dataclass
class TransformerConfigs:
    d_in: int = 37
    d_out: int = 29
    n_layers: int = 4
    d_model: int = 37  # Embedding dimension
    n_heads: int = 16  # Constraint: n_heads % d_model == 0
    sl: int = 1_000  # Sequence length
    mlp_scale: float = 4
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
            mlp_scale (float): Scaling factor for hidden dimension.
            d_model (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super(FFN, self).__init__()
        self.swiglu = SwiGLU(
            dim=configs.d_model, h_dim=int(configs.mlp_scale * configs.d_model),
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
            d_model (int): Input and output embedding dimension.
            mlp_scale (float): Scaling factor for hidden dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super().__init__()
        self.in_features = configs.d_model
        self.out_features = configs.d_model
        self.chunks = 2
        self.hidden_features = int(configs.mlp_scale * configs.d_model)

        self.fc_1 = nn.Linear(self.in_features, self.chunks * self.hidden_features, bias=configs.bias)
        self.fc_2 = nn.Linear(self.hidden_features, self.out_features, bias=configs.bias)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GatedFFN.

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

class TransformerBlock(nn.Module):
    """
    Single block of the Transformer.
    """

    def __init__(self, configs):
        super(TransformerBlock, self).__init__()
        self.configs = configs
        self.rn_1 = RMSNorm(configs.d_model, eps=configs.rms_norm_eps)
        self.rn_2 = RMSNorm(configs.d_model, eps=configs.rms_norm_eps)
        self.attn = self._get_attn_type(configs)

        self.ffn_1 = MoE(
            configs,
            experts=[GatedFFN(configs) for _ in range(configs.num_experts)],
            gate=nn.Linear(configs.d_model, configs.num_experts, bias=configs.bias)
        ) if configs.moe else GatedFFN(configs)

    def _get_attn_type(self, configs):
        if configs.dilated_attn:
            return DilatedCausalSelfAttention(configs)
        else:
            return CausalSelfAttention(configs)

    def forward(self, x):
        x = x + self.attn(self.rn_1(x))
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
        self.d_in = configs.d_in
        self.d_model = configs.d_model
        self.d_out = configs.d_out
        self.dropout = nn.Dropout(self.configs.dropout)
        
        if configs.moe:
            print(f"\nMoE?: Enabled | Using {configs.num_experts} experts.")
        else:
            print("\nMoE?: Disabled")

        self.transformer = nn.ModuleDict(
            dict(
                # Since our tasks are continuous, we do not use token embeddings.
                wpe=nn.Embedding(configs.sl, configs.d_model),
                dropout=self.dropout,
                hidden=nn.ModuleList(
                    [TransformerBlock(configs) for _ in range(configs.n_layers)]
                ),
            )
        )

        self.input_proj = nn.Linear(configs.d_in, self.d_model, bias=configs.bias)
        self.output_proj = nn.Linear(configs.d_model, self.d_out, bias=configs.bias)
        self.loss_fn = self.configs.loss_fn

        # Initialize all weights
        self.std = self.d_model**-0.5
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
        _, sl, _ = inputs.size()

        x = self.input_proj(inputs)

        # Generate positional embeddings for the sequence
        pos = torch.arange(0, sl, dtype=torch.long, device=inputs.device)  # -> (sl)

        # Position embeddings of shape (sl, d_model)
        pos_emb = self.transformer.wpe(pos)  # -> (sl, d_model)

        # Add positional embeddings to the input
        x = x + pos_emb

        incremental_state = None # 
        if self.configs.dilated_attn:
            incremental_state = {}

        x = self.transformer.dropout(x)

        # Pass through each transformer block in hidden layers
        for block in self.transformer.hidden:
            x = block(x)

        # Output model predictions
        preds = self.output_proj(x)

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
