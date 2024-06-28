# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Isabel Liu, Yagiz Devre
# File: model.py
# ==============================================================================#

"""Spectral temporal unit (STU) block."""

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
from tqdm import tqdm


@dataclass
class SSSMConfigs:
    d_in: int = 29
    d_out: int = 29
    n_layers: int = 6
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

    def apply_stu(self, inputs):
        eig_vals, eig_vecs = self.eigh
        x_tilde = compute_x_tilde(inputs, (eig_vals, eig_vecs))
        delta_phi = x_tilde @ self.m_phi
        delta_ar_u = compute_ar_x_preds(self.m_u, inputs)
        y_t = compute_y_t(self.m_y, delta_phi + delta_ar_u)
        return y_t

    def forward(self, inputs):
        eig_vals, eig_vecs = self.eigh
        x_tilde = compute_x_tilde(inputs, (eig_vals, eig_vecs))
        delta_phi = x_tilde @ self.m_phi
        delta_ar_u = compute_ar_x_preds(self.m_u, inputs)
        y_t = compute_y_t(self.m_y, delta_phi + delta_ar_u)
        return y_t


class FFN(nn.Module):
    """
    Simple feed-forward network.
    """

    def __init__(self, configs):
        super(FFN, self).__init__()
        self.h_dim = (configs.scale * configs.n_embd * 2) // 3
        # TODO: Consider implementing Squared ReLU from https://arxiv.org/pdf/2109.08668 ??
        self.swiglu = SwiGLU(dim=configs.n_embd, h_dim=self.h_dim, bias=configs.bias)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = self.swiglu(x)
        return self.dropout(x)


class SequentialBlock(nn.Module):
    def __init__(self, configs):
        super(SequentialBlock, self).__init__()
        self.n_embd = configs.n_embd
        self.d_out = configs.d_out
        self.n_layers = configs.n_layers
        self.dropout = configs.dropout
        self.sl = configs.sl
        self.num_eigh = configs.num_eigh
        self.k_u = configs.k_u
        self.k_y = configs.k_y
        self.learnable_m_y = configs.learnable_m_y
        self.ln_1 = nn.LayerNorm(self.n_embd, bias=configs.bias)
        self.stu = STU(configs)
        self.ln_2 = nn.LayerNorm(self.n_embd, bias=configs.bias)
        self.bn = nn.BatchNorm1d(self.sl)
        self.ffn = FFN(configs)
    #     self.m_ffn = nn.Parameter(torch.empty(self.n_embd))
    #     self.m_stu = nn.Parameter(torch.empty(self.n_embd))
    #     self._init_m_()

    # def _init_m_(self):
    #     nn.init.ones_(self.m_ffn) # TODO: 
    #     nn.init.ones_(self.m_stu)

    # TODO: Do last: try squared relu instead of FFN_SwiGLU
    # TODO: Normalize STU outputs (at all outputs)?
    # TODO: Gated residual connection post MLPs (does SiLU handle this?)
    def forward(self, x):
        z = x
        # x = self.stu(self.ln_1(x))
        # print(f"m_stu: {self.m_stu}")
        # x = self.ffn(self.m_stu(x))
        # print(f"m_ffn: {self.m_ffn}")
        # x = self.m_ffn * z + x
        # return x
        # z = x
        # x = self.stu(self.ln_1(x))
        # y = self.ffn(self.ln_2(z)) # TODO: Try concat and project back down
        # return x + y
        x = self.stu(self.ln_1(x))
        x = self.ffn(x)
        x = z + x
        return x
        

class SplitBlock(nn.Module):
    def __init__(self, configs):
        super(SequentialBlock, self).__init__()
        self.n_embd = configs.n_embd
        self.d_out = configs.d_out
        self.n_layers = configs.n_layers
        self.dropout = configs.dropout
        self.sl = configs.sl
        self.num_eigh = configs.num_eigh
        self.k_u = configs.k_u
        self.k_y = configs.k_y
        self.learnable_m_y = configs.learnable_m_y
        self.ln_1 = nn.LayerNorm(self.n_embd, bias=configs.bias)
        self.stu = STU(configs)
        self.ln_2 = nn.LayerNorm(self.n_embd, bias=configs.bias)
        self.bn = nn.BatchNorm1d(self.sl)
        self.ffn = FFN(configs)
    #     self.m_ffn = nn.Parameter(torch.empty(self.n_embd))
    #     self.m_stu = nn.Parameter(torch.empty(self.n_embd))
    #     self._init_m_()

    # def _init_m_(self):
    #     nn.init.ones_(self.m_ffn) # TODO: 
    #     nn.init.ones_(self.m_stu)

    # TODO: Do last: try squared relu instead of FFN_SwiGLU
    # TODO: Normalize STU outputs (at all outputs)?
    # TODO: Gated residual connection post MLPs (does SiLU handle this?)
    def forward(self, x):
        # TODO: Potential for MoE architecture here! Make it flaggable.
        # z = x
        # x = self.stu(self.ln_1(x))
        # print(f"m_stu: {self.m_stu}")
        # x = self.ffn(self.m_stu(x))
        # print(f"m_ffn: {self.m_ffn}")
        # x = self.m_ffn * z + x
        # return x
        z = x
        x = self.stu(self.ln_1(x))
        y = self.ffn(self.ln_2(z)) # TODO: Try concat and project back down
        return x + y


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
        self.task_head = nn.Linear(self.n_embd, self.d_out, bias=self.bias)

        self.emb = nn.Linear(self.n_embd, self.n_embd)
        self.stu = nn.ModuleDict(
            dict(
                # Since our tasks are continuous, we do not use token embeddings.
                wpe=nn.Embedding(self.sl, self.n_embd),
                dropout=nn.Dropout(self.dropout),
                hidden=nn.ModuleList([SequentialBlock(configs) for _ in range(self.n_layers)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=self.bias),
            )
        )
        self.projection = nn.Linear(self.d_in, self.d_out)

        if self.controls["task"] == "mujoco-v1":
            if self.controls["controller"] == "Ant-v1":
                self.d_out = 29
            else:
                self.d_out = 18

        # Initialize weights
        self.m_x = float(self.d_out) ** -0.5
        self.std = float(self.n_embd) ** -0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print("STU Model Parameter Count: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, mean=0.0, std=self.std)
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
        x = self.emb(inputs)
        bsz, sl, n_embd = inputs.size()

        # Generate positional embeddings for the sequence
        pos = torch.arange(0, sl, dtype=torch.long, device=inputs.device)  # -> (sl)
        pos_emb = self.stu.wpe(pos)

        # Add positional embeddings to input
        x = x + pos_emb.unsqueeze(0)
        x = self.stu.dropout(x)

        for block in self.stu.hidden:
            x = block(x)

        preds = self.projection(x)
        if self.controls["task"] != "mujoco-v3":
            loss, metrics = (
                self.loss_fn(preds, targets) if targets is not None else (None, None)
            )
            return preds, (loss, metrics)
        else:
            loss = self.loss_fn(preds, targets) if targets is not None else None
            return preds, (loss,)

    def predict(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        init: int = 0,
        t: int = 1,
    ) -> tuple[list[float], tuple[torch.Tensor, dict[str, float]]]:
        """
        Predicts the next states in trajectories and computes losses against the targets.

        Args:
            inputs (torch.Tensor): A tensor of shape [num_trajectories, seq_len, d_in].
            targets (torch.Tensor): A tensor of shape [num_trajectories, seq_len, d_out].
            init (int): The index of the initial state to start at.
            t (int): The number of time steps to predict.

        Returns:
            A tuple containing the list of predicted states after `t` time steps and
            a tuple containing the total loss and a dictionary of metrics.
        """
        device = inputs.device
        num_trajectories, seq_len, d_in = inputs.size()

        predicted_sequence = []
        total_loss = torch.tensor(0.0, device=device)
        metrics = {
            "loss": [],
            "coordinate_loss": [],
            "orientation_loss": [],
            "angle_loss": [],
            "coordinate_velocity_loss": [],
            "angular_velocity_loss": [],
        }

        for i in range(t):
            current_input_state = inputs[:, init + i, :].unsqueeze(1)
            current_target_state = targets[:, init + i, :].unsqueeze(1)

            # Predict the next state using the model
            next_state = self.model(current_input_state)
            loss, metric = self.loss_fn(next_state, current_target_state)

            predicted_sequence.append(next_state.squeeze(1).tolist())

            # Accumulate the metrics
            for key in metrics:
                metrics[key].append(metric[key])

            # Accumulate the losses
            total_loss += loss.item()

        total_loss /= t

        return predicted_sequence, (total_loss, metrics)
