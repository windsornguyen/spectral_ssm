# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: stu.py
# ==============================================================================#

"""The Spectral State Space Model architecture for benchmark tasks."""

import torch
import torch.nn as nn

from dataclasses import dataclass
from models.stu.stu_utils_old import (
    get_top_eigh,
    preconvolve,
    compute_ar_u,
    compute_spectral,
    compute_ar_y,
)
from utils.moe import MoE
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLU


@dataclass
class SpectralSSMConfigs:
    d_in: int = 10
    d_out: int = 10
    n_layers: int = 2
    d_model: int = 8
    sl: int = 1_000
    mlp_scale: int = 4
    embd_scale: int = 4
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
    task: str = "copy"
    vocab_size: int = 20
    loss_fn: nn.Module = nn.CrossEntropyLoss()
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
        self.k = configs.num_eigh
        self.use_ar_y = configs.use_ar_y
        self.use_ar_u = configs.use_ar_u
        self.sigma = sigma
        self.V = V  # Precomputed FFT of top K eigenvectors.
        self.padded_sl = padded_sl
        self.k_u = configs.k_u
        self.k_y = configs.k_y
        assert (self.k_u < self.k) or (self.k_y < self.k), "Cannot shift"
        self.learnable_m_y = configs.learnable_m_y
        self.stu_dropout = nn.Dropout(configs.dropout)
        self.resid_dropout = nn.Dropout(configs.dropout)

        # Parameterizable matrix Mᵘ, Mᵠ⁺, and Mᵠ⁻, per section 3
        self.M_u = nn.Parameter(torch.empty(self.k_u, self.d_model, self.d_model))
        self.M_phi_plus = nn.Parameter(torch.empty(self.k, self.d_model, self.d_model))
        self.M_phi_minus = nn.Parameter(torch.empty(self.k, self.d_model, self.d_model))

        # Parametrizable matrix Mʸ Introduced in section 5, equation 5
        if self.learnable_m_y:
            self.M_y = nn.Parameter(torch.zeros(self.d_model, self.k_y, self.d_model))
        else:
            self.register_buffer(
                "m_y", torch.zeros(self.d_model, self.k_y, self.d_model)
            )

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
                inputs,
                self.sigma,
                self.V,
                self.M_phi_plus,
                self.M_phi_minus,
                self.M_y,
                self.padded_sl,
                self.use_ar_y,
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
            mlp_scale (float): Scaling factor for MLP hidden dimension.
            d_model (int): Embedding dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs) -> None:
        super(MLP, self).__init__()
        self.h_dim = int(configs.mlp_scale * configs.d_model)
        self.swiglu = SwiGLU(
            dim=configs.d_model, h_dim=self.h_dim, bias=configs.bias, use_sq_relu=False
        )
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
            mlp_scale (float): Scaling factor for MLP hidden dimension.
            bias (bool): Whether to use bias in linear layers.
            dropout (float): Dropout rate.
    """

    def __init__(self, configs):
        super().__init__()
        self.in_features = configs.d_model
        self.out_features = configs.d_model
        self.hidden_features = int(configs.mlp_scale * configs.d_model)
        self.fc1 = nn.Linear(
            self.in_features, 2 * self.hidden_features, bias=configs.bias
        )
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


class ResidualSTU(nn.Module):
    def __init__(self, configs, num_models=3):
        super(ResidualSTU, self).__init__()
        self.configs = configs
        self.loss_fn = configs.loss_fn
        self.task = configs.task
        self.models = nn.ModuleList([SpectralSSM(configs) for _ in range(num_models)])
        self.num_models = num_models

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_preds = []
        all_losses = []

        # Initialize residual
        if self.task in ["copy", "induction", "associative"]:
            residual = torch.nn.functional.one_hot(
                targets, num_classes=self.configs.d_out
            ).float()
        else:
            residual = targets.unsqueeze(-1)  # Add dimension to match preds

        print(f"Inputs shape: {inputs.shape}")
        print(f"Residual shape: {residual.shape}")
        inputs = inputs.unsqueeze(-1).expand(-1, -1, self.configs.d_out)
        print(f"Reshaped inputs shape: {inputs.shape}")

        for i, model in enumerate(self.models):
            self.freeze_previous_models(i)
            preds, loss = model(inputs, residual)

            all_preds.append(preds)
            all_losses.append(loss)

            if loss.requires_grad:
                loss.backward(retain_graph=True)

            # Update residual based on the task
            if self.task in ["copy", "induction"]:
                # For classification, residual is the difference in probabilities
                residual = residual - torch.nn.functional.softmax(
                    preds.detach(), dim=-1
                )
            else:
                residual = residual - preds.detach()

        final_preds = sum(all_preds)
        # final_losses = sum(all_losses)

        if self.task == "copy":
            # Reshape logits to (bsz * sl, d_out) and targets to (bsz * sl)
            final_loss = self.loss_fn(
                final_preds.view(-1, final_preds.size(-1)), targets.view(-1)
            )
        elif self.task == "induction":
            # For induction, targets should already be of shape (bsz,)
            final_loss = self.loss_fn(final_preds[:, -1, :], targets)
        else:
            final_loss = self.loss_fn(final_preds.squeeze(), targets)

        self.unfreeze_all_models()
        # if final_loss.requires_grad:
        #     final_loss.backward(retain_graph=True)
        print("hey", final_preds.shape)
        return final_preds, final_loss

    def freeze_previous_models(self, current_model_index):
        for i in range(current_model_index):
            for param in self.models[i].parameters():
                param.requires_grad = False

    def unfreeze_all_models(self):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = True


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
        self.rn_1 = RMSNorm(configs.d_model)
        self.rn_2 = RMSNorm(configs.d_model)
        self.stu = STU(configs, sigma, V, padded_sl)
        self.mlp = (
            MoE(
                configs,
                experts=[GatedMLP(configs) for _ in range(configs.num_experts)],
                gate=nn.Linear(configs.d_model, configs.num_experts, bias=configs.bias),
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
        x = x + self.stu(self.rn_1(x))
        x = x + self.mlp(self.rn_2(x))
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
        self.sl = configs.sl
        self.num_eigh = configs.num_eigh
        self.learnable_m_y = configs.learnable_m_y
        self.alpha = configs.alpha
        self.use_hankel_L = configs.use_hankel_L
        self.device = configs.device

        self.sigma, self.phi = get_top_eigh(
            self.sl, self.num_eigh, self.use_hankel_L, self.device
        )
        self.V, self.padded_sl = preconvolve(self.phi, self.sl)  # Precomputed.

        self.bias = configs.bias
        self.dropout = configs.dropout
        self.loss_fn = configs.loss_fn
        self.task = configs.task

        self.spectral_ssm = nn.ModuleDict(
            dict(
                hidden=nn.ModuleList(
                    [
                        Block(self.configs, self.sigma, self.V, self.padded_sl)
                        for _ in range(self.n_layers)
                    ]
                ),
            )
        )

        # Add an embedding layer for the copying task
        if configs.task == "adding":
            self.embed = nn.Linear(configs.d_in, self.d_model)
        elif configs.task in ["copy", "induction", "associative"]:
            self.embed = nn.Embedding(configs.d_in, self.d_model)

        self.output = nn.Linear(self.d_model, configs.d_out, bias=self.bias)

        # Initialize all weights
        self.m_x = self.d_out**-0.5
        self.std = self.d_model**-0.5
        self.apply(self._init_weights)

        # Report the number of parameters
        print("\nSTU Model Parameter Count: %.4fM" % (self.get_num_params() / 1e6,))

    def forward(self, inputs, targets=None):
        """
        Forward pass of the spectral SSM model for the copying task.

        Args:
            inputs (torch.Tensor): Input tensor of shape (bsz, sl)
            targets (torch.Tensor, optional): Target tensor of shape (bsz, sl)

        Returns:
            tuple: (predictions, loss)
                - predictions: Output tensor of shape (bsz, sl, num_categories)
                - loss: Computed loss if targets are provided, else None
        """
        # Embed the input categories
        if self.task in ["copy", "induction", "associative"]:
            x = self.embed(inputs)  # Shape: (bsz, sl, d_model)
        elif self.task == "adding":
            # Reshape inputs from (bsz, sl * 2) to (bsz, sl, 2)
            x = inputs.view(inputs.shape[0], -1, self.configs.d_in)
            x = self.embed(x)  # Shape: (bsz, sl, d_model)

        # Apply the spectral SSM layers
        for layer in self.spectral_ssm.hidden:
            x = layer(x)

        # For associative recall, we only need to predict based on the last token
        if self.task == "associative":
            x = x[:, -1, :]  # Shape: (bsz, d_model)
            logits = self.output(x)  # Shape: (bsz, d_out)
        else:
            logits = self.output(x)  # Shape: (bsz, sl, d_out)

        # Compute predictions
        if self.task in ["copy", "induction", "associative"]:
            preds = torch.argmax(logits, dim=-1)
        elif self.task == "adding":
            preds = logits.mean(dim=1).squeeze(-1)

        if targets is not None:
            if self.task == "copy":
                loss = self.loss_fn(logits.view(-1, self.d_out), targets.view(-1))
            elif self.task == "induction":
                logits_flat = logits.view(
                    -1, logits.size(-1)
                )  # Shape: (bsz * sl, vocab_size)
                targets_flat = targets.view(-1)  # Shape: (bsz * sl)
                loss = self.loss_fn(logits_flat, targets_flat)
            elif self.task == "associative":
                loss = self.loss_fn(logits, targets)
            else:  # adding task
                loss = self.loss_fn(preds, targets)
            return preds, loss
        else:
            return preds, None

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
                    module.M_y[:, 1] = self.alpha * torch.eye(module.d_model)

    def get_num_params(self):
        """
        Return the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
