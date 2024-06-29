# =============================================================================#
# Authors: Windsor Nguyen
# File: stu_utils.py
# =============================================================================#

# TODO: Consider adding back @torch.jit.script to these functions.
"""Utilities for spectral SSM."""

import torch


def get_hankel(n: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Generates the special Hankel matrix Z, as defined in the paper.
    Note: This does not generate the Hankel matrix with the built-in
          negative featurization as mention in the appendix.

    Args:
        n (int): Size of the square Hankel matrix.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    i = torch.arange(1, n + 1)  # -> [n]
    s = i[:, None] + i[None, :]  # -> [n, n]
    Z = 2.0 / (s**3 - s + eps)  # -> [n, n]
    return Z


def get_hankel_L(n: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Generates the special Hankel matrix Z_L, as defined in the appendix.
    Offers negative featurization.

    Args:
        n (int): Size of the square Hankel matrix.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    # TODO: Not yet vectorized or broadcastable.
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")

    # Calculate (-1)^(i+j-2) + 1
    sgn = (-1) ** (i + j - 2) + 1

    # Calculate the denominator
    denom = (i + j + 3) * (i + j - 1) * (i + j + 1)

    # Combine all terms
    Z_L = sgn * (8 / (denom + eps))

    return Z_L


def get_top_eigh(
    n: int, K: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top K eigenvalues and eigenvectors of the Hankel matrix Z.

    Args:
        n (int): Size of the Hankel matrix.
        K (int): Number of top eigenvalues/eigenvectors to return.
        device (torch.device): Computation device (CPU/GPU).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - sigma: Top K eigenvalues [K]
            - phi: The corresponding eigenvectors [n, K]
    """
    Z = get_hankel(n).to(device)  # -> [n, n]
    sigma, phi = torch.linalg.eigh(Z)  # -> [n], [n, n]
    return sigma[-K:], phi[:, -K:]  # -> [k, (n, k)]


def next_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than or equal to x.

    Args:
        x (int): The input integer.

    Returns:
        int: The smallest power of 2 greater than or equal to x.
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def conv(phi: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute convolution to project input sequences into the spectral basis.

    This implements the computation of U⁺_{t,k} and U⁻_{t,k}, as described
    in Section 3 of the paper.

    Args:
        phi (torch.Tensor): Top K eigenvectors of shape [sl, K].
        u (torch.Tensor): Input of shape [bsz, sl, d_in].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two feature tensors of shape [bsz, sl, K, d_in].
    """
    # Extract dims
    bsz, sl, d_in = u.shape
    K = phi.shape[1]

    # Round sequence length to the nearest power of 2 for efficient convolution
    n = next_power_of_2(sl * 2 - 1)

    # Expand phi and u to the return shape
    phi = phi.view(1, -1, K, 1).expand(bsz, -1, K, d_in)
    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    # Compute U⁺
    V = torch.fft.rfft(phi, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    U_plus = torch.fft.irfft(V * U, n=n, dim=1)[:, :sl]

    # Generate alternating signs tensor, (-1)^i of length sl, match dims of u
    alt = torch.ones(sl, device=u.device)
    alt[1::2] = -1  # Replace every other element with -1, starting from index 1
    alt = alt.view(1, sl, 1, 1).expand_as(u)

    # Compute U⁻
    u_alt = u * alt
    U_alt = torch.fft.rfft(u_alt, n=n, dim=1)
    U_minus = torch.fft.irfft(V * U_alt, n=n, dim=1)[:, :sl]

    return U_plus, U_minus


def compute_U(
    inputs: torch.Tensor, eigh: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Compute the x_tilde component of spectral state space model.

    Args:
        inputs (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): A tuple of eigenvalues of shape [K,] and
            eigenvectors of shape [sl, K].

    Returns:
        torch.Tensor: x_tilde: A tensor of shape [bsz, sl, K, d_in].
    """
    # Project inputs into the spectral basis
    return conv(eigh[1], inputs)  # -> tuple of [bsz, sl, k, d_in]


def compute_y_t(M_y, y):
    bsz, sl, d_out = y.shape
    k_y = M_y.shape[0]
    expanded_y = y.unsqueeze(1).expand(bsz, k_y, sl, d_out)
    reshaped_y = expanded_y.reshape(bsz * k_y, sl, d_out)
    repeated_M_y = M_y.transpose(-1, -2).repeat(
        bsz, 1, 1
    )  # TODO: Why is this transpose necessary?

    o = torch.bmm(reshaped_y, repeated_M_y)
    o = o.view(bsz, k_y, sl, d_out)
    yt = torch.zeros_like(y)
    for k in range(k_y):
        yt[:, k + 1 :] += o[:, k, : sl - k - 1]
    return yt


def compute_u_t(M_u, u):
    bsz, sl, d_in = u.shape
    k_u = M_u.shape[0]
    expanded_u = u.unsqueeze(1).expand(bsz, k_u, sl, d_in)
    reshaped_u = expanded_u.reshape(bsz * k_u, sl, d_in)
    repeated_M_u = M_u.repeat(bsz, 1, 1)

    o = torch.bmm(reshaped_u, repeated_M_u)
    o = o.view(bsz, k_u, sl, -1)
    ut = torch.zeros(bsz, sl, o.shape[-1], device=u.device)
    for t in range(k_u):
        ut[:, t+1:] += o[:, t, : sl - t - 1]
    return ut


def compute_ar(
    M_y: torch.Tensor, M_u: torch.Tensor, y: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    """
    to be written.
    """
    # Sum M^y_i \hat_{y}_{t-i} from i=1 to i=k_y
    yt = compute_y_t(M_y, y)

    # Sum M^u_i \hat_{u}_{t+1-i} from i=1 to i=k_y + 1
    ut = compute_u_t(M_u, u)

    return yt + ut


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Shift time axis by k steps to align u_{t-k} with u_t.

    This function implements the time shifting functionality needed for
    the autoregressive component in Equation 4 of the STU model (Section 3).

    Args:
        u (torch.Tensor): An input tensor of shape [bsz, sl, K, d],
        where bsz is the batch size, sl is the sequence length,
        K is the number of spectral filters, and d is the feature dimension.
        k (int): Number of time steps to shift. Defaults to 1.

    Returns:
        torch.Tensor: Shifted tensor of shape [bsz, sl, K, d].
    """
    if k == 0:
        return u

    # Extract dims
    bsz, sl, K, d = u.shape

    # Pad k time steps
    padding = torch.zeros(bsz, k, K, d, device=u.device)  # -> [bsz, k, K, d]

    if k < sl:
        # Prepend and truncate last k sequences
        shifted = torch.cat([padding, u[:, :-k]], dim=1)  # -> [bsz, sl, K, d]
    else:
        shifted = padding[:, :sl]

    return shifted


def compute_spectral(
    inputs: torch.Tensor,
    eigh: tuple[torch.Tensor, torch.Tensor],
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    # Spectral component
    # 1. Sum from k=1 to K m_phi_plus at k, times sigma ** 0.25 at k, times U⁺ at t-i, k where i <= k_y
    # 2. Sum from k=1 to K m_phi_minus at k, times sigma ** 0.25 at k, times U⁻ at t-i, k where i <= k_y
    # 3. Sum the two terms above together to get the spectral component
    sigma, phi = eigh
    K = phi.shape[1]
    U_plus_tilde, U_minus_tilde = compute_U(inputs, eigh)
    U_shifted = torch.stack(
        [shift(U_plus_tilde, k_y), shift(U_minus_tilde, k_y)], dim=-1
    )

    # Combine m_phi matrices
    sigma_root = sigma.pow(0.25).view(1, 1, K, 1, 1)
    U_weighted = U_shifted * sigma_root

    m_phi = torch.stack([m_phi_plus, m_phi_minus], dim=-1)

    # Compute the spectral component in a single einsum
    # TODO: Make this faster (no einsum)
    result = torch.einsum("bskid,kiod->bso", U_weighted, m_phi)

    return result
