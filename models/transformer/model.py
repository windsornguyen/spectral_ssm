# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: (Transformer) model.py
#
# Full definition of a standard Transformer model adapted for regression
# on physics-based trajectories with
#
# 1). Flash Attention,
# 2). Memory-Efficient Attention, and
# 3). Dilated Attention
#
# References:
# 1) the official GPT-2 TensorFlow implementation released by OpenAI:
# https://github.com/openai/gpt-2/blob/master/src/model.py
#
# 2) huggingface/transformers PyTorch implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
# =============================================================================#

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange
from torch.nn import functional as F
from tqdm import tqdm
from utils.squared_relu import SquaredReLU
from utils.rms_norm import RMSNorm # TODO: Remove the bias config since we don't use LN

class CausalSelfAttention(nn.Module):
    """
    Self-attention layer for the Transformer.

    Note: scaled_dot_product_attention enables FlashAttention-2
    (Tri Dao, 2023, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")
    and Memory-Efficient Attention (Rabe et al., 2022, "Self-attention Does Not Need O(n^2) Memory"),
    all written in C++, per the PyTorch documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """

    def __init__(self, configs):
        super(CausalSelfAttention, self).__init__()
        assert configs.n_embd % configs.n_head == 0

        # Key, query, value projections for all heads, concatenated
        self.c_attn = nn.Linear(configs.n_embd, 3 * configs.n_embd, bias=configs.bias)

        # The output projection, concatenated
        self.c_proj = nn.Linear(configs.n_embd, configs.n_embd, bias=configs.bias)
        self.c_proj.SCALE_INIT = 1

        # Regularization
        self.dropout = configs.dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.n_embd = configs.n_embd
        self.n_head = configs.n_head

        # Flash attention makes the GPUs go brrr, but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # Manual implementation of the causal mask
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(configs.sl, configs.sl)).view(
                    1, 1, configs.sl, configs.sl
                ),
            )

    def forward(self, x):
        """
        Performs the forward pass of the causal self attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (bsz, sl, n_embd), where bsz is the batch size,
                sl is the sequence length, and n_embd is the embedding dimensionality (n_embd).

        Returns:
            torch.Tensor: Output tensor of shape (bsz, sl, n_embd) after applying self-attention.
        """
        bsz, sl, n_embd = x.size()

        # Compute query, key, values for all heads in batch, and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(bsz, sl, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # -> (B, nh, sl, hs)
        q = q.view(bsz, sl, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, sl, hs)
        v = v.view(bsz, sl, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (B, nh, sl, hs)

        # Causal self-attention; self-attend: (bsz, nh, sl, hs) x (bsz, nh, hs, sl) -> (B, nh, sl, sl)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual implementation of self-attention
            q = q * k.size(-1) ** -0.5
            att = q @ k.transpose(-2, -1)
            att = att.masked_fill(self.mask[:, :, :sl, :sl] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (bsz, nh, sl, sl) x (bsz, nh, sl, hs) -> (bsz, nh, sl, hs)

        # Re-assemble / "concat" all attention head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(bsz, sl, n_embd)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FFN(nn.Module):
    """
    Simple feed-forward network with the squared ReLU activation function.
    """

    def __init__(self, configs):
        super(FFN, self).__init__()
        self.c_fc = nn.Linear(configs.n_embd, configs.scale * configs.n_embd, bias=configs.bias)
        self.squared_relu = SquaredReLU()
        self.c_proj = nn.Linear(configs.scale * configs.n_embd, configs.n_embd, bias=configs.bias)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.squared_relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single block of the Transformer.
    """

    def __init__(self, configs):
        super(TransformerBlock, self).__init__()
        self.rn_1 = RMSNorm(configs.n_embd)
        self.attn = self._get_attn_type(configs)
        self.rn_2 = RMSNorm(configs.n_embd)
        self.ffn = FFN(configs)

    def _get_attn_type(self, configs):
        if configs.use_dilated_attn:
            return DilatedCausalSelfAttention(configs)
        else:
            return CausalSelfAttention(configs)

    def forward(self, x):
        z = x
        x = x + self.attn(self.rn_1(x))
        x = x + self.ffn(self.rn_2(x))
        return x + z


@dataclass
class TransformerConfigs:
    n_layers: int = 12
    n_embd: int = 512  # Embedding dimension
    n_head: int = 16  # Constraint: n_head % n_embd == 0
    sl: int = 300  # Sequence length
    scale: int = 4
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.10
    use_dilated_attn: bool = False
    loss_fn: nn.Module = nn.MSELoss()
    controls: dict = field(
        default_factory=lambda: {"task": "mujoco-v3", "controller": "Ant-v1"}
    )


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
        self.transformer = nn.ModuleDict(
            dict(
                # Since our tasks are continuous, we do not use token embeddings.
                wpe=nn.Embedding(configs.sl, configs.n_embd),
                dropout=nn.Dropout(configs.dropout),
                hidden=nn.ModuleList(
                    [TransformerBlock(configs) for _ in range(configs.n_layers)]
                ),
                ln_f=nn.LayerNorm(configs.n_embd, bias=configs.bias),
            )
        )

        # Adjust output dims based on task and controller
        self.d_out = configs.n_embd
        if configs.controls["task"] == "mujoco-v1":
            if configs.controls["controller"] == "Ant-v1":
                self.d_out = 29
            else:
                self.d_out = 18

        self.task_head = nn.Linear(configs.n_embd, self.d_out, bias=configs.bias)
        self.loss_fn = self.configs.loss_fn

        # Initialize all weights
        self.std = self.n_embd ** -0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print(
            "Transformer Model Parameter Count (excl. pos. emb.): %.2fM"
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
        device = inputs.device
        bsz, sl, d_in = inputs.size()

        # Generate positional embeddings for the sequence
        pos = torch.arange(0, sl, dtype=torch.long, device=device)  # -> (sl)

        # Position embeddings of shape (sl, n_embd)
        pos_emb = self.transformer.wpe(pos)  # -> (sl, n_embd)

        # Add bsz dim to pos_emb
        pos_emb = pos_emb.unsqueeze(0).expand(bsz, -1, -1)

        # Project input to lower-dimensional space
        x = self.d_in(inputs)  # -> (bsz, sl, n_embd)

        # Apply dropout
        x = self.transformer.dropout(x) + self.transformer.dropout(pos_emb)

        # Pass through each transformer block in hidden layers
        for block in self.transformer.hidden:
            x = block(x)

        # Output model predictions!
        preds = self.task_head(x)  # -> (bsz, sl, d_out)

        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            return preds, (loss,)

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
            cfg.n_head,
            cfg.d_embd // cfg.n_head,
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
        preds = torch.zeros(num_trajectories, steps, self.configs.n_embd, device=device)
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

                # for key in metrics:
                #     metrics[key][:, i] = step_metrics[key]

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
                window_start = max(0, i - self.configs.sl + 1)

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


class DilatedCausalSelfAttention(CausalSelfAttention):
    """
    Dilated causal self-attention layer, as implemented in the LongNet paper
    (Ding et al., 2023, "LongNet: Scaling Transformers to 1,000,000,000 Tokens").

    This code was adapted from torchscale/component/dilated_attention.py.
    The repository can be found at https://github.com/microsoft/torchscale.
    """

    def dense_to_sparse(self, x, ratio):
        # Get the length of the sequence
        length = x.size(1)

        # Calculate padding needed for sequence length and number of heads to be multiples of ratio
        padding = (ratio - length % ratio) % ratio
        head_padding = (ratio - self.n_head % ratio) % ratio

        # Apply padding if needed
        if padding > 0 or head_padding > 0:
            x = F.pad(x, (0, 0, 0, head_padding, 0, padding), value=0.0)

        # Rearrange tensor to apply dilated attention
        x = rearrange(x, "b (l r1) (r2 h) d -> b l h d r1 r2", r1=ratio, r2=ratio)
        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        x = rearrange(x, "b l h d r -> b l (r h) d")

        # Remove extra padding from heads
        if head_padding > 0:
            x = x[:, :, : self.n_head]

        return x

    def sparse_to_dense(self, out, lse, ratio):
        # Calculate padding needed for number of heads to be a multiple of ratio
        head_padding = (ratio - self.n_head % ratio) % ratio

        # Apply padding if needed
        if head_padding > 0:
            out = F.pad(out, (0, 0, 0, head_padding), value=0.0)
            lse = F.pad(lse, (0, 0, 0, head_padding), value=-1e8)

        # Rearrange tensor to convert back from sparse to dense representation
        out = rearrange(out, "b l (r h) d -> b l h d r", r=ratio)
        out = torch.diag_embed(out, offset=0, dim1=4, dim2=5)
        out = rearrange(out, "b l h d r1 r2 -> b (r2 h) (l r1) d", r1=ratio, r2=ratio)

        # Handle logsumexp for sparse to dense conversion
        lse = rearrange(lse, "b (r h) l -> b l h r", r=ratio)
        lse = torch.diag_embed(lse, offset=0, dim1=3, dim2=4)
        lse = lse.masked_fill_(lse == 0, -1e8)
        lse = rearrange(lse, "b l h r1 r2 -> b (r2 h) (l r1) 1", r1=ratio, r2=ratio)

        # Remove extra padding from heads
        if head_padding > 0:
            out = out[:, : self.n_head]
            lse = lse[:, : self.n_head]

        return out, lse

    def gather_kv(self, x, sl, seq_len, is_causal=True):
        # Get batch size
        bsz = x.size(0)

        # Ensure segment length is a multiple of sequence length
        assert sl % seq_len == 0
        num_rank_per_segment = sl // seq_len

        # Gather all key-value pairs from different ranks
        x = all_gather_func(x)
        current_rank = get_data_parallel_rank()
        x = rearrange(x, "(w b) l h d -> w b l h d", b=bsz)

        # Apply causal masking if needed
        if is_causal:
            if current_rank > 0:
                x = x[:current_rank]
            else:
                x = x[:1] * 0

        # Get current segment based on rank
        current_segment = current_rank // num_rank_per_segment * num_rank_per_segment
        x = x[current_segment : current_segment + num_rank_per_segment]

        # Rearrange tensor to combine segments
        x = rearrange(x, "w b l h d -> b (w l) h d")
        return x

    def gathering(
        self, x, dr, sl, is_causal=True, offset=0, is_kv=False, seq_parall=True
    ):
        curr_x = x

        # Apply padding if offset is greater than zero
        if offset > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, offset % sl, 0), value=0.0)

        # Get sequence length
        seq_len = curr_x.size(1)

        # Determine if key-value pairs should be gathered based on sequence parallelism
        should_gather_kv = is_kv and seq_parall and (sl > seq_len)
        _sl = sl
        sl = min(sl, seq_len)
        padding = (sl - seq_len % sl) % sl

        # Apply padding if needed
        if padding > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, 0, padding), value=0.0)

        # Rearrange tensor for dilated attention
        curr_x = rearrange(curr_x, "b (n g) h d -> (b n) g h d", g=sl)
        curr_x = self.dense_to_sparse(curr_x, dr)

        # Gather key-value pairs if needed
        if should_gather_kv:
            curr_x = self.gather_kv(curr_x, _sl, seq_len, is_causal)

        # Rearrange tensor for attention computation
        curr_x = rearrange(curr_x, "b l h d -> (b h) l d")

        return curr_x

    # TODO: Initialize the dilation ratios to what the paper used.
    def scattering(self, outs, lses, seq_len, bsz, offset=0):
        assert len(outs) == len(lses)
        assert len(outs) % len(self.args.dilated_ratio) == 0
        all_outs, all_lses = [], []
        drs = (
            self.args.dilated_ratio
        )  # TODO: (Dynamically) replace with actual dilation ratios
        if len(outs) > len(drs):
            drs = drs * (len(outs) // len(drs))

        for dr, o, lse in zip(drs, outs, lses, strict=True):
            o = rearrange(o, "b l (h d) -> b l h d", h=self.n_head)
            o, lse = self.sparse_to_dense(o, lse, dr)
            o = rearrange(o, "(b n) h g d -> (b h) (n g) d", b=bsz)
            lse = rearrange(lse, "(b n) h g 1 -> (b h) (n g) 1", b=bsz)
            o = o[:, offset : offset + seq_len]
            lse = lse[:, offset : offset + seq_len]
            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0).max(0)[0]
            all_lses = [torch.exp(lse - max_lse) for lse in all_lses]
            lse_sum = torch.stack(all_lses, dim=0).sum(0)
            all_lses = [lse / lse_sum for lse in all_lses]

        out = sum(o * lse.type_as(o) for o, lse in zip(all_outs, all_lses, strict=True))
        out = rearrange(out, "(b h) l d -> b l (h d)", h=self.n_head)
        return out

    def forward(self, x):
        # Get batch size, sequence length, and embedding dimension
        B, T, C = x.size()

        # Compute query, key, and value projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Initialize lists for storing outputs and logsumexp results
        outs, lses = [], []

        # Replace with actual segment lengths and dilation ratios
        for sl, dr in zip([128], [1], strict=True):
            # Gather key, value, and query tensors
            ki = self.gathering(k, dr, sl, is_causal=True, is_kv=True, seq_parall=True)
            vi = self.gathering(v, dr, sl, is_causal=True, is_kv=True, seq_parall=True)
            qi = self.gathering(q, dr, sl, is_causal=True, is_kv=False, seq_parall=True)

            if self.flash:
                out, lse = torch.nn.functional.scaled_dot_product_attention(
                    qi,
                    ki,
                    vi,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
            else:
                att = (qi @ ki.transpose(-2, -1)) * (1.0 / math.sqrt(ki.size(-1)))
                att = att.masked_fill(self.mask[:, :, :sl, :sl] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                out = att @ vi  # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, hs)

            outs.append(out)
            lses.append(lse)

        # Scatter outputs and logsumexp results
        y = self.scattering(outs, lses, T, B)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # -> (B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
