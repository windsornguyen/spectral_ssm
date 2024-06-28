# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha, Evan Dogariu
# File: stu_utils.py
# =============================================================================#

"""Utilities for spectral SSM."""

import torch


# TODO: Moved to models/stu class defn
def get_hankel_matrix(n: int) -> torch.Tensor:
    """
    Generate a spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.

    Returns:
        torch.Tensor: A spectral Hankel matrix of shape [n, n].
    """
    indices = torch.arange(1, n + 1)  # -> [n]
    sums = indices[:, None] + indices[None, :]  # -> [n, n]
    z = 2.0 / (sums**3 - sums)  # -> [n, n]
    return z


def get_top_hankel_eigh(
    n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.
        k (int): Number of eigenvalues to return.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of eigenvalues of shape [k,] and
            eigenvectors of shape [n, k].
    """
    hankel_matrix = get_hankel_matrix(n).to(device)  # -> [n, n]
    eig_vals, eig_vecs = torch.linalg.eigh(hankel_matrix)  # -> [n], [n, n]
    return eig_vals[-k:], eig_vecs[:, -k:]  # -> [k, (n, k)]


def get_random_real_matrix(
    shape: list[int],
    scaling: float,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """
    Generate a random real matrix.

    Args:
        shape (list[int]): Shape of the matrix to generate.
        scaling (float): Scaling factor for the matrix values.
        lower (float, optional): Lower bound of truncated normal distribution
            before scaling.
        upper (float, optional): Upper bound of truncated normal distribution
            before scaling.

    Returns:
        torch.Tensor: A random real matrix scaled by the specified factor.
    """
    random_matrix = torch.randn(shape)
    clamped_matrix = torch.clamp(random_matrix, min=lower, max=upper)
    return scaling * clamped_matrix


# TODO: Do shape analysis on this function


def shift(x: torch.Tensor) -> torch.Tensor:
    """
    Shift time axis by one to align x_{t-1} with x_t.

    Simulates a time shift where each timestep of input tensor
    is moved one step forward in time, and initial timestep is replaced w/ zeros.

    Args:
        x (torch.Tensor): A tensor of shape [sl, d], where 'sl' is the
                          sequence length and 'd' is the feature dimension.

    Returns:
        torch.Tensor: A tensor of the same shape as 'x' where the first timestep
                      is zeros and each subsequent timestep 'i' contains values
                      from timestep 'i-1' of the input tensor.
    """
    # Construct a zero tensor for the initial timestep
    init_step = torch.zeros_like(x[:1])

    # Remove the last timestep
    remaining_steps = x[:-1]

    # Concat the initial zero tensor with the remaining sliced tensor
    shifted = torch.cat([init_step, remaining_steps], dim=0)

    return shifted


# TODO: Put this somewhere else?
def nearest_power_of_2(x: int):
    s = bin(x)
    s = s.lstrip("-0b")
    length = len(s)
    return 1 << (length - 1) if x == 1 << (length - 1) else 1 << length


def conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis.

    Args:
        v (torch.Tensor): Top k eigenvectors of shape [sl, k].
        u (torch.Tensor): Input of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: A matrix of shape [bsz, sl, k, d_in].
    """
    # bsz, sl, d_in = u.shape
    # _, k = v.shape

    # # Round n to the nearest power of 2
    # n = nearest_power_of_2(sl * 2 - 1)

    # # Add and expand the bsz and d_in dims in v
    # v = v.view(1, sl, k, 1) # -> [1, sl, k, 1]
    # v = v.expand(bsz, sl, k, d_in) # -> [bsz, sl, k, d_in]

    # # Add and expand the k dim in u
    # u = u.view(bsz, sl, 1, d_in) # -> [bsz, sl, 1, d_in]
    # u = u.expand(bsz, sl, k, d_in) # -> [bsz, sl, k, d_in]

    # # Perform convolution!
    # V = torch.fft.rfft(v, n=n, dim=1)
    # U = torch.fft.rfft(u, n=n, dim=1)
    # Z = V * U
    # z = torch.fft.irfft(Z, n=n, dim=1)

    # return z[:, :sl]
    bsz, sl, d_in = u.shape
    k = v.shape[1]

    # Round n to the nearest power of 2
    n = nearest_power_of_2(sl * 2 - 1)

    # Add and expand the bsz and d_in dims in v
    v = v.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, d_in)

    # Add and expand the k dim in u
    u = u.unsqueeze(2).expand(-1, -1, k, -1)

    # Perform convolution!
    V = torch.fft.rfft(v, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    Z = V * U
    z = torch.fft.irfft(Z, n=n, dim=1)

    return z[:, :sl]


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


def compute_ar_x_preds(m_u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the auto-regressive component of spectral SSM.

    Args:
        m_u (torch.Tensor): A weight matrix of shape [d_out, d_in, k].
        x (torch.Tensor): Batch of input sequences of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: ar_x_preds: An output of shape [bsz, sl, d_out].
    """
    bsz, sl, d_in = x.shape
    d_out, _, k = m_u.shape

    # Contract over `d_in` to combine weights with input sequences
    o = torch.einsum("oik,bli->bklo", m_u, x)  # [bsz, k, l, d_out]

    # For each `i` in `k`, shift outputs by `i` positions to align for summation.
    rolled_o = torch.stack(
        [torch.roll(o[:, i], shifts=i, dims=1) for i in range(k)], dim=1
    )  # -> [bsz, k, l, d_out]

    # Create a mask that zeros out nothing at `k=0`, the first `(sl, d_out)` at
    # `k=1`, the first two `(sl, dout)`s at `k=2`, etc.
    mask = torch.triu(torch.ones((k, sl), device=m_u.device))
    mask = mask.view(k, sl, 1)  # Add d_out dim: -> [k, sl, 1]

    # Apply the mask and sum along `k`
    return torch.sum(rolled_o * mask, dim=1)


def compute_x_tilde(
    inputs: torch.Tensor, eigh: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Compute the x_tilde component of spectral state space model.

    Args:
        inputs (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): A tuple of eigenvalues of shape [k,] and
            eigenvectors of shape [sl, k].

    Returns:
        torch.Tensor: x_tilde: A tensor of shape [bsz, sl, k * d_in].
    """
    eig_vals, eig_vecs = eigh
    k = eig_vals.size(0)
    bsz, sl, d_in = inputs.shape

    # Project inputs into the spectral basis
    x_spectral = conv(eig_vecs, inputs)  # -> [bsz, sl, k, d_in]

    # Reshape dims of eig_vals to match dims of x_spectral
    eig_vals = eig_vals.view(1, 1, k, 1)  # -> [1, 1, k, 1]

    # Perform spectral filtering on x to obtain x_tilde
    x_tilde = x_spectral * eig_vals**0.25

    # TODO: May have to adjust this once we introduce autoregressive component.
    # Reshape x_tilde so that it's matmul-compatible with m_phi
    return x_tilde.view(bsz, sl, k * d_in)
