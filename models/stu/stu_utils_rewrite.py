# =============================================================================#
# Authors: Windsor Nguyen
# File: stu_utils.py
# =============================================================================#

"""Utility functions for spectral SSM."""

import torch


@torch.jit.script
def get_hankel(n: int) -> torch.Tensor:
    """
    Generates a Hankel matrix Z, as defined in the paper.

    Note: This does not generate the Hankel matrix with the built-in
    negative featurization as mentioned in the appendix.

    Args:
        n (int): Size of the square Hankel matrix.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    i = torch.arange(1, n + 1)         # -> [n]
    s = i[:, None] + i[None, :]        # -> [n, n]
    Z = 2.0 / (s**3 - s)               # -> [n, n]
    return Z


@torch.jit.script
def get_hankel_L(n: int) -> torch.Tensor:
    """
    Generates an alternative Hankel matrix Z_L that offers built-in
    negative featurization as mentioned in the appendix.

    Args:
        n (int): Size of the square Hankel matrix.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    i = torch.arange(1, n + 1)
    s = i[:, None] + i[None, :]        # s = i + j
    sgn = (-1) ** (s - 2) + 1
    denom = (s + 3) * (s - 1) * (s + 1)
    Z_L = sgn * (8 / denom)
    return Z_L


@torch.jit.script
def get_top_eigh(
    n: int, K: int, use_hankel_L: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the top K eigenvalues and eigenvectors of the Hankel matrix Z.

    Args:
        n (int): Size of the Hankel matrix.
        K (int): Number of top eigenvalues/eigenvectors to return.
        device (torch.device): Computation device (CPU/GPU).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - sigma: Top K eigenvalues [K]
            - phi: The corresponding eigenvectors [n, K]
    """
    Z = get_hankel_L(n).to(device) if use_hankel_L else get_hankel(n).to(device)
    sigma, phi = torch.linalg.eigh(Z)  # -> [n], [n, n]
    return sigma[-K:], phi[:, -K:]     # -> [K], [n, K]


@torch.jit.script
def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Roll the time axis forward by k steps to align the input u_{t-k} with u_t.
    This erases the last k time steps of the input tensor.

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
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted


@torch.jit.script
def compute_ar(
    M_y: torch.Tensor, y: torch.Tensor, M_u: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    """
    Computes the full AR-STU (Auto-Regressive Spectral Transform Unit) model output.

    This function combines the autoregressive component (past outputs) and the input component
    to produce the final output of the AR-STU model.

    Args:
        y (torch.Tensor): Past output tensor of shape (bsz, sl, d_out)
        u (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
        M_y (torch.Tensor): Output weight matrices of shape (k_y, d_out, d_out)
        M_u (torch.Tensor): Input weight matrices of shape (k_u, d_in, d_out)
        k_y (int): Number of past outputs to consider

    Returns:
        torch.Tensor: Full AR-STU model output of shape (bsz, sl, d_out)

    Note:
        bsz: Batch size
        sl: Sequence length
        d_in: Input dimension
        d_out: Output dimension
        k_u: Number of past inputs to consider (inferred from M_u shape)
    """
    k_y, k_u = M_y.shape[0], M_u.shape[0]

    # Sum M^y_i \hat_{y}_{t-i} from i=1 to i=k_y
    y_shifts = torch.stack([shift(y, i + 1) for i in range(k_y)], dim=1)
    ar_y = torch.einsum("bksd,kod->bso", y_shifts, M_y)

    # Sum M^u_i \hat_{u}_{t+1-i} from i=1 to i=k_u
    u_shifts = torch.stack([shift(u, i) for i in range(k_u)], dim=1)
    ar_u = torch.einsum("bksd,kdi->bsi", u_shifts, M_u)

    return ar_y + ar_u


@torch.jit.script
def nearest_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than or equal to x. 
    If x is already a power of 2, it returns x itself.
    Otherwise, it returns the next higher power of 2.

    Args:
        x (int): The input integer.

    Returns:
        int: The smallest power of 2 that is greater than or equal to x.
    """
    s = bin(x)
    s = s.lstrip("-0b")
    length = len(s)
    return 1 << (length - 1) if x == 1 << (length - 1) else 1 << length


@torch.jit.script
def conv(u: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FFT convolution of the input sequences into the Hankel spectral basis.

    This implements the computation of U⁺_{t,k} and U⁻_{t,k}, as described
    in Section 3 of the paper.

    Args:
        u (torch.Tensor): Input of shape [bsz, sl, d].
        phi (torch.Tensor): Top K eigenvectors of shape [sl, K].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Feature tensors of shape [bsz, sl, K, d].
    """
    bsz, sl, d = u.shape
    _, K = phi.shape

    # Round sequence length to the nearest power of 2 for efficient convolution
    n = nearest_power_of_2(sl * 2 - 1)

    # Add bsz and d dims to phi and u and expand to the return shape
    phi = phi.view(1, -1, K, 1).expand(bsz, -1, K, d)
    u = u.view(bsz, -1, 1, d).expand(bsz, -1, K, d)

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


@torch.jit.script
def compute_spectral(
    inputs: torch.Tensor,
    eigh: tuple[torch.Tensor, torch.Tensor],
    M_phi_plus: torch.Tensor,
    M_phi_minus: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    """
    Computes the spectral component of AR-STU U feature vectors by projecting the input
    tensor into the spectral basis via convolution.

    Args:
        inputs (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): A tuple of eigenvalues of shape [K,] and
            eigenvectors of shape [sl, K].
        m_phi_plus (torch.Tensor): A tensor of shape [K, d_in, d_out].
        m_phi_minus (torch.Tensor): A tensor of shape [K, d_in, d_out].
        k_y (int): Number of time steps to shift.

    Returns:
        torch.Tensor: The spectral component tensor of shape [bsz, sl, d_out].
    """
    sigma, phi = eigh
    _, K = phi.shape

    # Compute U⁺ and U⁻
    U_plus, U_minus = conv(inputs, phi) # -> tuple of [bsz, sl, K, d_in]

    # Shift U⁺ and U⁻ k_y time steps
    _U_plus = shift(U_plus, k_y)
    _U_minus = shift(U_minus, k_y)

    # Perform spectral filter on U⁺ and U⁻ w/ sigma
    sigma_root = (sigma ** 0.25).view(1, 1, K, 1)
    U_plus_filtered, U_minus_filtered = _U_plus * sigma_root, _U_minus * sigma_root
    
    # Sum M^{\phi +}_k \cdot U_plus_filtered across K filters
    spectral_plus = torch.einsum("bsKd,Kdo->bso", U_plus_filtered, M_phi_plus)

    # Sum M^{\phi -}_k \cdot U_minus_filtered across K filters
    spectral_minus = torch.einsum("bsKd,Kdo->bso", U_minus_filtered, M_phi_minus)

    return spectral_plus + spectral_minus
