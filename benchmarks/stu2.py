# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: stu2 (STU-Attention Hybrid but with linear attention).py
# ==============================================================================#

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from utils.rms_norm import RMSNorm
from spectral_ssm.models.stu.stu_utils_old import (
    get_top_eigh,
    preconvolve,
)
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
    def __init__(self, configs, sigma, phi, padded_sl) -> None:
        super(STU, self).__init__()
        self.configs = configs
        self.d_in = configs.d_in
        self.d_model = configs.d_model
        self.d_out = configs.d_out
        self.k = configs.num_eigh
        self.use_ar_y = configs.use_ar_y
        self.use_ar_u = configs.use_ar_u
        self.sigma = sigma
        self.phi = phi  # Precomputed FFT of top K eigenvectors.
        self.padded_sl = padded_sl
        self.k_u = configs.k_u
        self.k_y = configs.k_y

        # Parameterizable matrix Mᵘ, Mᵠ⁺, and Mᵠ⁻, per section 3
        self.Q = nn.Parameter(
            torch.empty(self.k, self.d_model, self.d_model)
        )
        self.K = nn.Parameter(
            torch.empty(self.k, self.d_model, self.d_model)
        )
        self.V = nn.Parameter(
            torch.empty(self.k, self.d_model, self.d_model)
        )

        # self.c_proj = nn.Linear(configs.d_model, configs.d_out, bias=configs.bias)
        # self.c_proj.SCALE_INIT = 1

    @torch.jit.script
    def featurize(
        u: torch.Tensor, V: torch.Tensor, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the FFT convolution of the input sequences into the Hankel
        spectral basis, as described in Section 3 of the paper.

        This function computes U⁺_{t,k} and U⁻_{t,k}, which are the positive and
        negative featurizations of the input sequence, respectively.

        Args:
            u (torch.Tensor): Input of shape [B, L, D].
            V (torch.Tensor): Precomputed FFT of top K eigenvectors of shape [1, n//2+1, K, 1].
            n (int): Padded sequence length for FFT.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature tensors U⁺ and U⁻ of shape [B, L, K, D].
        """
        B, L, D = u.shape
        K = V.size(2)

        # Prepare u for both positive and negative convolutions
        u_plus = u.view(B, L, 1, D).expand(B, L, K, D)
        u_minus = u_plus.clone()
        u_minus[:, 1::2] *= -1

        # Compute U⁺ and U⁻
        U_plus = torch.fft.rfft(u_plus, n=n, dim=1)
        U_minus = torch.fft.rfft(u_minus, n=n, dim=1)

        # Perform the convolutions!
        U_plus = torch.fft.irfft(V * U_plus, n=n, dim=1)[:, :L]
        U_minus = torch.fft.irfft(V * U_minus, n=n, dim=1)[:, :L]

        return U_plus, U_minus

    def spectral_filter(self, U1, U2, U3, M_phi1, M_phi2, M_phi3):
        """
        Apply spectral filtering to the featurized inputs.

        Args:
            U1, U2, U3: [B: batch, L: sequence length, K: filters, D: input dimension]
            M_phi1, M_phi2, M_phi3: [K: filters, D_out: output dimension, D: input dimension]

        Returns:
            Three tensors of shape [B: batch, L: sequence length, D_out: output dimension]
        """
        sigma_root = (self.sigma**0.25).view(1, 1, -1, 1)
        U1_filtered = U1 * sigma_root
        U2_filtered = U2 * sigma_root
        U3_filtered = U3 * sigma_root

        out1 = torch.einsum("blkd,kod->blo", U1_filtered, M_phi1)
        out2 = torch.einsum("blkd,kod->blo", U2_filtered, M_phi2)
        out3 = torch.einsum("blkd,kod->blo", U3_filtered, M_phi3)

        return out1, out2, out3

    def stu_attention(self, Q, K, V) -> torch.Tensor:
        Z = torch.einsum("bsp,bsn->bspn", V, K)
        H = torch.cumsum(Z, dim=1)
        Y = torch.einsum("btn,btpn->btp", Q, H)
        return Y

    def kernel(self, x):
        return F.elu(x)+1
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.size()

        # Featurize input sequence with all heads
        q_BLKD, k_BLKD = self.featurize(u, self.phi, self.padded_sl)
        
        v_BLKD = q_BLKD + k_BLKD

        # Apply spectral filtering to get Q, K, V
        q_BLD, k_BLD, v_BLD = self.spectral_filter(
            q_BLKD, k_BLKD, v_BLKD, self.Q, self.K, self.V
        )

        q_BLD = self.kernel(q_BLD)
        k_BLD = self.kernel(k_BLD)
        # v_BLKD = F.gelu(q_BLKD + k_BLKD)
        # q_BLKD = self.kernel(q_BLKD)
        # k_BLKD = self.kernel(k_BLKD)
        

        # Analyze q, k, v tensors
        self.analyze_qkv(q_BLD, k_BLD, v_BLD)

        # Compute attention scores
        y_BLD = self.stu_attention(q_BLD, k_BLD, v_BLD)

        return y_BLD

    def analyze_qkv(self, q, k, v):
        """Analyze the characteristics of q, k, and v tensors."""
        # Compute cosine similarity between q, k, and v
        q_k_sim = self.cosine_similarity(q, k)
        q_v_sim = self.cosine_similarity(q, v)
        k_v_sim = self.cosine_similarity(k, v)

        print(f"Average cosine similarity - Q-K: {q_k_sim:.4f}, Q-V: {q_v_sim:.4f}, K-V: {k_v_sim:.4f}")

        # Compute entropy of attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        entropy = self.compute_entropy(attn_weights)

        print(f"Average attention entropy: {entropy:.4f}")

        # Compute rank of Q, K, V matrices
        q_rank = self.matrix_rank(q.view(-1, q.size(-1)))
        k_rank = self.matrix_rank(k.view(-1, k.size(-1)))
        v_rank = self.matrix_rank(v.view(-1, v.size(-1)))

        print(f"Average rank - Q: {q_rank:.2f}, K: {k_rank:.2f}, V: {v_rank:.2f}")

        # Compute condition number of Q, K, V matrices
        q_cond = self.condition_number(q.view(-1, q.size(-1)))
        k_cond = self.condition_number(k.view(-1, k.size(-1)))
        v_cond = self.condition_number(v.view(-1, v.size(-1)))

        print(f"Average condition number - Q: {q_cond:.2f}, K: {k_cond:.2f}, V: {v_cond:.2f}")

    def cosine_similarity(self, x, y):
        """Compute average cosine similarity between two tensors."""
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        sim = (x_norm * y_norm).sum(dim=-1).mean().item()
        return sim

    def compute_entropy(self, attn_weights):
        """Compute the entropy of attention weights."""
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)
        return entropy.mean().item()

    def matrix_rank(self, matrix):
        """Compute the average rank of a batch of matrices."""
        singular_values = torch.svd(matrix).S
        ranks = torch.sum(singular_values > 1e-5, dim=-1).float().mean().item()
        return ranks

    def condition_number(self, matrix):
        """Compute the average condition number of a batch of matrices."""
        singular_values = torch.svd(matrix).S
        condition_numbers = singular_values.max(dim=-1)[0] / (singular_values.min(dim=-1)[0] + 1e-9)
        return condition_numbers.mean().item()

    def test_robustness(self, input_tensor, num_perturbations=10, perturbation_scale=0.01):
        """Test the robustness of the model to small input perturbations."""
        original_output = self.forward(input_tensor)
        
        perturbation_outputs = []
        for _ in range(num_perturbations):
            perturbed_input = input_tensor + torch.randn_like(input_tensor) * perturbation_scale
            perturbed_output = self.forward(perturbed_input)
            perturbation_outputs.append(perturbed_output)

        output_variations = torch.stack([torch.abs(po - original_output).mean() for po in perturbation_outputs])
        avg_variation = output_variations.mean().item()
        max_variation = output_variations.max().item()

        print("Robustness test results:")
        print(f"Average output variation: {avg_variation:.6f}")
        print(f"Maximum output variation: {max_variation:.6f}")

        return avg_variation, max_variation


class GatedMLP(nn.Module):
    """
    Gated multi-layer perceptron network using SiLU activation.

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

        self.fc1 = nn.Linear(
            self.in_features, self.chunks * self.hidden_features, bias=configs.bias
        )
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=configs.bias)
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
        y = self.fc1(x)
        y, gate = y.chunk(self.chunks, dim=-1)
        y = y * self.silu(gate)
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
        self.rn_1 = RMSNorm(configs.d_model)
        self.rn_2 = RMSNorm(configs.d_model)
        self.stu = STU(configs, sigma, V, padded_sl)
        self.mlp = GatedMLP(configs)
        # self.mlp = SwiGLU(
        #     dim=configs.d_model,
        #     h_dim=configs.d_model,
        #     bias=configs.bias,
        #     use_sq_relu=False,
        # )

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
        # x = x + self.stu(x)
        # x = x + self.mlp(x)
        # return x


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
        # Embed the input categories
        if self.configs.task in ["copy", "induction", "associative"]:
            x = self.embed(inputs)  # Shape: (bsz, sl, d_model)
        elif self.configs.task == "adding":
            # Reshape inputs from (bsz, sl * 2) to (bsz, sl, 2)
            x = inputs.view(inputs.shape[0], -1, self.configs.d_in)
            x = self.embed(x)  # Shape: (bsz, sl, d_model)

        # Apply the spectral SSM layers
        for layer in self.spectral_ssm.hidden:
            x = layer(x)

        # For associative recall, we only need to predict based on the last token
        if self.configs.task == "associative":
            x = x[:, -1, :]  # Shape: (bsz, d_model)
            logits = self.output(x)  # Shape: (bsz, d_out)
        else:
            logits = self.output(x)  # Shape: (bsz, sl, d_out)

        # Compute predictions
        if self.configs.task in ["copy", "induction", "associative"]:
            preds = torch.argmax(logits, dim=-1)
        elif self.configs.task == "adding":
            preds = logits.mean(dim=1).squeeze(-1)

        if targets is not None:
            if self.configs.task == "copy":
                loss = self.loss_fn(logits.view(-1, self.d_out), targets.view(-1))
            elif self.configs.task == "induction":
                logits_flat = logits.view(
                    -1, logits.size(-1)
                )  # Shape: (bsz * sl, vocab_size)
                targets_flat = targets.view(-1)  # Shape: (bsz * sl)
                loss = self.loss_fn(logits_flat, targets_flat)
            elif self.configs.task == "associative":
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
            if hasattr(module, "SCALE_INIT"):
                # Scale by 2 to account for self-attn and ffn sub-layer
                self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            torch.nn.init.xavier_normal_(module.Q)
            torch.nn.init.xavier_normal_(module.K)
            torch.nn.init.xavier_normal_(module.V)

    def get_num_params(self):
        """
        Return the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
