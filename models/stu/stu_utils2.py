# =============================================================================#
# Authors: Windsor Nguyen
# File: stu_utils.py
# =============================================================================#

# TODO: Consider adding back @torch.jit.script to these functions.
"""Utilities for spectral SSM."""

import torch


def get_rand_matrix(
    shape: list[int],
    scale: float,
    lo: float = -2.0,
    hi: float = 2.0,
) -> torch.Tensor:
    """
    Generate a random real matrix with truncated normal distribution.

    Args:
        shape (list[int]): Dimensions of the matrix.
        scale (float): Scaling factor for the matrix values.
        lo (float): Lower bound of truncation (before scaling).
        hi (float): Upper bound of truncation (before scaling).

    Returns:
        torch.Tensor: Random matrix of shape 'shape', values in
                      range [lo * scale, hi * scale].
    """
    M = torch.randn(shape)
    M_clamp = torch.clamp(M, min=lo, max=hi)
    return scale * M_clamp


def get_hankel(n: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Generate the special Hankel matrix Z, as defined in the paper.

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
    phi = phi.view(1, sl, K, 1).expand(bsz, sl, K, d_in)
    u = u.view(bsz, sl, 1, d_in).expand(bsz, sl, K, d_in)

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


def compute_y_t(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Computes a sequence of y_t given a series of deltas and a transition matrix m_y.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [bsz, sl, d_out].

    Returns:
        torch.Tensor: A matrix of shape [bsz, sl, d_out].
    """
    d_out, k, _ = m_y.shape
    bsz, sl, _ = deltas.shape

    # Define the transition matrix A, and add bsz for bmm
    A = m_y.view(d_out, k * d_out)  # Reshape m_y to [d_out, k * d_out] for concat
    eye = torch.eye(
        (k - 1) * d_out, k * d_out, dtype=deltas.dtype, device=deltas.device
    )
    A = torch.cat([A, eye], dim=0)
    A = A.unsqueeze(0).expand(
        bsz, k * d_out, k * d_out
    )  # -> [bsz, k * d_out, k * d_out]

    # Add (k - 1) rows of padding to deltas
    padding = torch.zeros(
        bsz, sl, (k - 1) * d_out, dtype=deltas.dtype, device=deltas.device
    )  # -> [bsz, sl, (k - 1) * d_out]

    carry = torch.cat([deltas, padding], dim=2)  # -> [bsz, sl, k * d_out]

    # Reshape for sequential processing
    carry = carry.view(bsz, sl, k * d_out, 1)  # -> [bsz, sl, k * d_out, 1]

    # Initialize y and the output list of y's
    y = carry[:, 0]  # -> [bsz, k * d_out, 1]
    ys = [y[:, :d_out, 0]]  # -> [bsz, d_out]

    # Iterate through the sequence
    # TODO: Unsure of how to further vectorize/optimize this given its sequential nature.
    # This loop takes up __98%__ of this function.
    for i in range(1, sl):
        y = torch.bmm(A, y) + carry[:, i]
        ys.append(y[:, :d_out, 0])
    ys = torch.stack(ys, dim=1)  # -> [bsz, sl, d_out]

    return ys


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Shift time axis by k steps to align u_{t-k} with u_t.

    This function implements the time shift needed for the autoregressive
    component in Equation 4 of the STU model (Section 3).

    Args:
        u (torch.Tensor): An input tensor of shape [bsz, sl, d],
        where bsz is the batch size, sl is the sequence length,
        and d is the feature dimension.
        k (int): Number of time steps to shift. Defaults to 1.

    Returns:
        torch.Tensor: Shifted tensor of shape [bsz, sl, d].
    """
    if k == 0:
        return u

    # Extract dims
    bsz, sl, d = u.shape

    # Pad k time steps
    padding = torch.zeros(bsz, k, d, device=u.device)  # -> [bsz, k, d]

    if k < sl:
        # Prepend and truncate last k sequences
        shifted = torch.cat([padding, u[:, :-k]], dim=1)  # -> [bsz, sl, d]
    else:
        shifted = padding[:, :sl]

    return shifted


def compute_ar_x_preds(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the auto-regressive component of spectral SSM.

    Args:
        w (torch.Tensor): A weight matrix of shape [d_out, d_in, k].
        x (torch.Tensor): Batch of input sequences of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: ar_x_preds: An output of shape [bsz, sl, d_out].
    """
    bsz, sl, d_in = x.shape
    d_out, _, k = w.shape

    # Contract over `d_in` to combine weights with input sequences
    o = torch.einsum("oik,bli->bklo", w, x)  # [bsz, k, l, d_out]

    # For each `i` in `k`, shift outputs by `i` positions to align for summation.
    rolled_o = torch.stack(
        [torch.roll(o[:, i], shifts=i, dims=1) for i in range(k)], dim=1
    )  # -> [bsz, k, l, d_out]

    # Create a mask that zeros out nothing at `k=0`, the first `(sl, d_out)` at
    # `k=1`, the first two `(sl, d_out)`s at `k=2`, etc.
    mask = torch.triu(torch.ones((k, sl), device=w.device))
    mask = mask.view(k, sl, 1)  # Add d_out dim: -> [k, sl, 1]

    # Apply the mask and sum along `k`
    return torch.sum(rolled_o * mask, dim=1)


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
    eig_vals, eig_vecs = eigh
    k = eig_vals.size(0)
    bsz, sl, d_in = inputs.shape

    # Project inputs into the spectral basis
    U_plus, U_minus = conv(eig_vecs, inputs)  # -> tuple of [bsz, sl, k, d_in]

    # Reshape dims of eig_vals to match dims of U⁺ and U⁻
    eig_vals = eig_vals.view(1, 1, k, 1)  # -> [1, 1, k, 1]

    # Perform spectral filtering
    U_plus_tilde, U_minus_tilde = U_plus * eig_vals**0.25, U_minus * eig_vals**0.25

    return U_plus_tilde, U_minus_tilde


def compute_ar(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor
) -> torch.Tensor:
    """
    to be written.
    """
    bsz, sl, d_out = y.shape
    k_y, k_u = m_y.shape[0], m_u.shape[0]

    ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    # Sum M^y_i \hat_{y}_{t-i} from i=1 to i=k_y
    for i in range(k_y):
        y_shifted = shift(y, i + 1)
        ar_component += torch.einsum("btd,od->bto", y_shifted, m_y[i])

    # Sum M^u_i \hat_{u}_{t+1-i} from i=1 to i=k_y + 1
    for i in range(k_u):
        u_shifted = shift(u, i)
        ar_component += torch.einsum("btd,di->bti", u_shifted, m_u[i])

    return ar_component


def compute_spectral(
    U_plus_tilde: torch.Tensor,
    U_minus_tilde: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    sigma: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    # Spectral component
    # 1. Sum from k=1 to K m_phi_plus at k, times sigma ** 0.25 at k, times U⁺ at t-i, k where i <= k_y
    # 2. Sum from k=1 to K m_phi_minus at k, times sigma ** 0.25 at k, times U⁻ at t-i, k where i <= k_y
    # 3. Sum the two terms above together to get the spectral component

    U_shifted = torch.stack([shift(U_plus_tilde, k_y), shift(U_minus_tilde, k_y)], dim=-1)

    # Compute σ^(1/4) and combine operations
    sigma_root = (sigma**0.25).view(1, 1, -1, 1, 1)
    U_weighted = U_shifted * sigma_root

    # Combine m_phi matrices
    m_phi = torch.stack([m_phi_plus, m_phi_minus], dim=-1)

    # Compute the spectral component in a single einsum
    result = torch.einsum("bskid,kiod->bso", U_weighted, m_phi)

    return result


def apply_stu(
    u: torch.Tensor,
    eigh: tuple[torch.Tensor, torch.Tensor],
    m_y: torch.Tensor,
    m_u: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    # Sum compute_ar and compute_spectral together.
    bsz, sl, d_in = u.shape
    d_out = m_y.shape[0]

    # Initialize y with zeros
    ys = torch.zeros(bsz, sl, d_out, device=u.device)

    # Compute x_tilde
    U_plus_tilde, U_minus_tilde = compute_U(u, eigh)

    for t in range(sl):
        # Compute autoregressive component
        ar = compute_ar(ys[:, :t], u[:, : t + 1], m_y, m_u, min(t, k_y))

        # Compute spectral component
        spectral = compute_spectral(
            U_plus_tilde[:, : t + 1],
            U_minus_tilde[:, : t + 1],
            m_phi_plus,
            m_phi_minus,
            min(t + 1, k_y),
        )

        # Combine components
        ys[:, t] = spectral[:, -1] + ar[:, -1]

    return ys
