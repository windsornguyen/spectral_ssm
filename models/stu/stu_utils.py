# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: stu_utils.py
# =============================================================================#

"""Utility functions for spectral SSM."""

import torch
from utils.nearest_power_of_2 import nearest_power_of_2


@torch.jit.script
def get_hankel(n: int) -> torch.Tensor:
    """
    Generates a Hankel matrix Z, as defined in Equation (3) of the paper.

    This special matrix is used for the spectral filtering in the Spectral
    Transform Unit (STU).

    Args:
        n (int): Size of the square Hankel matrix.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    i = torch.arange(1, n + 1)
    s = i[:, None] + i[None, :]
    Z = 2.0 / (s**3 - s)
    return Z


@torch.jit.script
def get_hankel_L(n: int) -> torch.Tensor:
    """
    Generates an alternative Hankel matrix Z_L as defined in Equation (7) 
    of the paper's appendix.
    
    This version offers built-in negative featurization and can be used as
    an alternative to the standard Hankel matrix for spectral filtering.

    Args:
        n (int): Size of the square Hankel matrix.

    Returns:
        torch.Tensor: Hankel matrix Z_L of shape [n, n].
    """
    i = torch.arange(1, n + 1)
    s = i[:, None] + i[None, :]  # s = i + j
    sgn = (-1.0) ** (s - 2.0) + 1.0
    denom = (s + 3.0) * (s - 1.0) * (s + 1.0)
    Z_L = sgn * (8.0 / denom)
    return Z_L


@torch.jit.script
def get_top_eigh(
    n: int, K: int, use_hankel_L: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the top K eigenvalues and eigenvectors of the Hankel matrix Z.
    
    These eigenvalues and eigenvectors are used to construct the spectral
    filters for the STU model, as described in Section 3 of the paper.

    Args:
        n (int): Size of the Hankel matrix.
        K (int): Number of top eigenvalues/eigenvectors to return.
        use_hankel_L (bool): If True, use the alternative Hankel matrix Z_L.
        device (torch.device): Computation device (CPU/GPU).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - sigma: Top K eigenvalues [K]
            - phi: The corresponding eigenvectors [n, K]
    """
    Z = get_hankel_L(n).to(device) if use_hankel_L else get_hankel(n).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    return sigma[-K:], phi[:, -K:]


@torch.jit.script
def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Rolls the time axis forward by k steps to align the input u_{t-k} with u_t.
    This effectively removes the last k time steps of the input tensor.

    This function implements the time shifting functionality needed for
    the autoregressive component in Equation 4 of the STU model (Section 3).

    Args:
        u (torch.Tensor): An input tensor of shape [bsz, sl, K, d].
        k (int): Number of time steps to shift. Defaults to 1.

    Returns:
        torch.Tensor: Shifted tensor of shape [bsz, sl, K, d].
    """
    if k == 0:
        return u
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted


@torch.jit.script
def compute_ar_u(M_u: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
    """
    Computes the autoregressive component of the STU model with respect to
    the input, as described in Equation (4) of Section 3.

    This function implements the sum of M^u_i u_{t+1-i} from i=1 to 
    (more generally) k_u (in the paper, it was up until i=3).

    Args:
        M_u (torch.Tensor): Input weight matrices of shape (k_u, d_out, d_in)
        u_t (torch.Tensor): Input tensor of shape (bsz, sl, d_in)

    Returns:
        torch.Tensor: Autoregressive component w.r.t. input of shape (bsz, sl, d_out)
    """
    k_u = M_u.shape[0]

    # TODO: Do we shift by another parameter or by k_u? i.e. Do we shift only if use_ar_u?
    # Sum M^u_i \hat_{u}_{t+1-i} from i=1 to i=k_u
    u_shifted = torch.stack([shift(u_t, i) for i in range(k_u)], dim=1)
    ar_u = torch.einsum("bksi,koi->bso", u_shifted, M_u)

    return ar_u


@torch.jit.script
def compute_ar_y(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """
    Computes the autoregressive component of the AR-STU model with respect to
    the output, as described in Equation (6) of Section 5.

    This function implements the sum of M^y_i y_{t-i} from i=1 to i=k_y.
    It can be optimized further by using a scanning algorithm.

    Args:
        M_y: Transition weight matrices of shape (d_out, k_y, d_out)
        y_t: Predictions at current time step (bsz, sl, d_out)

    Returns:
        torch.Tensor: Autoregressive component w.r.t. output of shape (bsz, sl, d_out)
    
    Visualization:

    (1). Transition matrix A and its effect:
    Matrix A                      Input y_t           Output y_t+1
    +---------------------+       +---------+         +---------+
    | M_y1   M_y2   M_y3  |       | y_t     |         | y_t+1   |
    |  I      0      0    |   ×   | y_t-1   |    =    | y_t     |
    |  0      I      0    |       | y_t-2   |         | y_t-1   |
    |  0      0      I    |       | y_t-3   |         | y_t-2   |
    +---------------------+       +---------+         +---------+
    
    (2). State structure with padding:
    +---------+
    | y_t     | Current input.
    |  0      |
    |  0      | Preallocated (k - 1) rows for previous states.
    |  0      |
    +---------+
    
    (3). Computation for each time step:
    Matrix A                    Current state y_t    New input y_next    Output y_{t+1}
    +---------------------+     +--------------+     +---------------+   +--------------+
    | M_y1   M_y2   M_y3  |     | y_t          |     | y_{t+1}       |   | y_{t+1}      |
    |  I      0      0    |  ×  | y_{t-1}      |  ⊕  | 0             | = | y_t          |
    |  0      I      0    |     | y_{t-2}      |     | 0             |   | y_{t-1}      |
    |  0      0      I    |     | y_{t-3}      |     | 0             |   | y_{t-2}      |
    +---------------------+     +--------------+     +---------------+   +--------------+
    """
    d_out, k_y, _ = M_y.shape
    bsz, sl, _ = y_t.shape

    # (1). Construct transition matrix A:
    #      Identity has (k - 1) rows => [(k - 1) * d_out, k * d_out]
    eye = torch.eye((k_y - 1) * d_out, k_y * d_out, dtype=y_t.dtype, device=y_t.device)
    A = M_y.reshape(d_out, k_y * d_out) # <-- Ensure matmul-compatible with eye
    A = torch.cat([A, eye], dim=0)      # <-- Stack A atop the identity matrices
    A = A.unsqueeze(0).expand(bsz, k_y * d_out, k_y * d_out) # <-- Add bsz dim for bmm

    # (2). Prepare state with padding
    padding = torch.zeros(bsz, sl, (k_y - 1) * d_out, dtype=y_t.dtype, device=y_t.device)
    state = torch.cat([y_t, padding], dim=2)    # -> [bsz, sl, k * d_out]
    state = state.view(bsz, sl, k_y * d_out, 1) # Reshape for sequential processing

    # Initialize the first y_t and list of outputs
    y = state[:, 0]         # -> [bsz, k * d_out, 1]
    ys = [y[:, :d_out, 0]]  # -> [bsz, d_out]

    # (3). Iterate through the sequence length (starting from the 2nd time step)
    for i in range(1, sl):
        y_next = state[:, i]
        y = torch.bmm(A, y) + y_next
        ys.append(y[:, :d_out, 0])

    return torch.stack(ys, dim=1)


@torch.jit.script
def preconvolve(phi: torch.Tensor, sl: int) -> tuple[torch.Tensor, int]:
    """
    Precomputes the FFT of phi for use in the conv function.

    Args:
        phi (torch.Tensor): Top K eigenvectors of shape [sl, K].
        sl (int): Original sequence length.

    Returns:
        tuple[torch.Tensor, int]: Precomputed FFT of phi, padded sequence length.
    """
    _, K = phi.shape
    n = nearest_power_of_2(sl * 2 - 1)
    phi = phi.view(1, -1, K, 1)
    return torch.fft.rfft(phi, n=n, dim=1), n

@torch.jit.script
def conv(u: torch.Tensor, V: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the FFT convolution of the input sequences into the Hankel 
    spectral basis, as described in Section 3 of the paper.

    This function computes U⁺_{t,k} and U⁻_{t,k}, which are the positive and
    negative featurizations of the input sequence, respectively.

    Args:
        u (torch.Tensor): Input of shape [bsz, sl, d].
        V (torch.Tensor): Precomputed FFT of top K eigenvectors of shape [1, n//2+1, K, 1].
        n (int): Padded sequence length for FFT.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Feature tensors U⁺ and U⁻ of shape [bsz, sl, K, d].
    """
    bsz, sl, d = u.shape
    _, _, K, _ = V.shape

    # Prepare u for both positive and negative convolutions
    u_plus = u.view(bsz, sl, 1, d).expand(bsz, sl, K, d)
    u_minus = u_plus.clone()
    u_minus[:, 1::2] *= -1

    # Compute U⁺ and U⁻
    U_plus = torch.fft.rfft(u_plus, n=n, dim=1)
    U_minus = torch.fft.rfft(u_minus, n=n, dim=1)

    # Perform the convolutions!
    U_plus = torch.fft.irfft(V * U_plus, n=n, dim=1)[:, :sl]
    U_minus = torch.fft.irfft(V * U_minus, n=n, dim=1)[:, :sl]

    return U_plus, U_minus


@torch.jit.script
def compute_spectral(
    inputs: torch.Tensor,
    sigma: torch.Tensor,
    V: torch.Tensor,
    M_phi_plus: torch.Tensor,
    M_phi_minus: torch.Tensor,
    M_y: torch.Tensor,
    n: int,
    use_ar_y: bool
) -> torch.Tensor:
    """
    Computes the spectral component of the STU or AR-STU model, as described
    in Equations (4) and (6) of the paper.

    This function projects the input tensor into the spectral basis via 
    convolution and applies the precomputed spectral filters.

    Args:
        inputs (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        sigma (torch.Tensor): Eigenvalues tensor of shape [K,].
        V (torch.Tensor): Precomputed FFT of top K eigenvectors of shape [1, n//2+1, K, 1].
        M_phi_plus (torch.Tensor): Positive spectral filter weights [K, d_out, d_in].
        M_phi_minus (torch.Tensor): Negative spectral filter weights [K, d_out, d_in].
        M_y (torch.Tensor): Autoregressive weights for AR-STU [d_out, k_y, d_out].
        n (int): Padded sequence length for FFT.
        use_ar_y (bool): Whether to use autoregressive on the output sequence.

    Returns:
        torch.Tensor: The spectral component tensor of shape [bsz, sl, d_out].
    """
    k_y = M_y.shape[1]
    K = M_phi_plus.shape[0]

    # Compute U⁺ and U⁻
    U_plus, U_minus = conv(inputs, V, n)  # -> tuple of [bsz, sl, K, d_in]

    # Shift U⁺ and U⁻ k_y time steps
    if use_ar_y: # TODO: Do we shift by another parameter or by k_y? i.e. Do we shift only if use_ar_y?
        U_plus, U_minus = shift(U_plus, k_y), shift(U_minus, k_y)

    # Perform spectral filter on U⁺ and U⁻ w/ sigma
    sigma_root = (sigma**0.25).view(1, 1, K, 1)
    U_plus_filtered, U_minus_filtered = U_plus * sigma_root, U_minus * sigma_root

    # Sum M^{\phi +}_k \cdot U_plus_filtered across K filters
    spectral_plus = torch.einsum("bsKi,Koi->bso", U_plus_filtered, M_phi_plus)

    # Sum M^{\phi -}_k \cdot U_minus_filtered across K filters
    spectral_minus = torch.einsum("bsKi,Koi->bso", U_minus_filtered, M_phi_minus)

    return spectral_plus + spectral_minus
