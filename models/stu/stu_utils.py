# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
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
    i = torch.arange(1, n + 1)  # -> [n]
    s = i[:, None] + i[None, :]  # -> [n, n]
    Z = 2.0 / (s**3 - s)  # -> [n, n]
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
    s = i[:, None] + i[None, :]  # s = i + j
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
    return sigma[-K:], phi[:, -K:]  # -> [K], [n, K]


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
        K is the number of spectral filters, andp d is the feature dimension.
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
def compute_ar_u(M_u: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Computes the full AR-STU (Auto-Regressive Spectral Transform Unit) model output.

    This function combines the autoregressive component (past outputs) and the input component
    to produce the final output of the AR-STU model.

    Args:
        M_u (torch.Tensor): Input weight matrices of shape (k_u, d_out, d_in)
        u (torch.Tensor): Input tensor of shape (bsz, sl, d_in)

    Returns:
        torch.Tensor: Full AR-STU model output of shape (bsz, sl, d_out)

    Note:
        bsz: Batch size
        sl: Sequence length
        d_in: Input dimension
        d_out: Output dimension
        k_u: Number of past inputs to consider (inferred from M_u shape)
    """
    k_u = M_u.shape[0]

    # Sum M^u_i \hat_{u}_{t+1-i} from i=1 to i=k_u
    u_shifted = torch.stack([shift(u, i) for i in range(k_u)], dim=1)
    ar_u = torch.einsum("bksd,kod->bso", u_shifted, M_u)

    return ar_u


@torch.jit.script
def compute_ar_y(M_y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the auto-regressive component of spectral SSM.

    Args:
        M_y: Output weight matrices of shape (k_y, d_out, d_out)
        y: Predictions (bsz, sl, d_out)

    Returns:
        torch.Tensor: AR component of shape (bsz, sl, d_out)
    """
    k_y, d_out, _ = M_y.shape
    bsz, sl, _ = y.shape

    # Define the transition matrix A, and add bsz for bmm
    A = M_y.reshape(k_y * d_out, d_out)    # Reshape M_y to [k * d_out, d_out] for concat
    eye = torch.eye(k_y * d_out, (k_y - 1) * d_out, dtype=y.dtype, device=y.device)
    A = torch.cat([A, eye], dim=1)
    A = A.unsqueeze(0).expand(bsz, k_y * d_out, k_y * d_out)

    # Add (k_y - 1) rows of padding to y
    padding = torch.zeros(bsz, sl, (k_y - 1) * d_out, dtype=y.dtype, device=y.device)   # -> [bsz, sl, (k_y - 1) * d_out]

    carry = torch.cat([y, padding], dim=2)  # -> [bsz, sl, k_y * d_out]

    # Reshape for sequential processing
    carry = carry.view(bsz, sl, k_y * d_out, 1) # -> [bsz, sl, k_y * d_out, 1]

    # Initialize y and the output list of y's
    y_t = carry[:, 0] # -> [bsz, k_y * d_out, 1]
    ar_y = [y_t[:, :d_out, 0]]  # ->[bsz, d_out]

    # Iterate through the sequence
    for i in range(1, sl):
        y_t = torch.bmm(A, y_t) + carry[:, i]
        ar_y.append(y_t[:, :d_out, 0])
    ar_y = torch.stack(ar_y, dim=1) # -> [bsz, sl, d_out]

    return ar_y


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
    M_y: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes the spectral component of AR-STU U feature vectors by projecting the input
    tensor into the spectral basis via convolution.

    Args:
        inputs (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): A tuple of eigenvalues of shape [K,] and
            eigenvectors of shape [sl, K].
        m_phi_plus (torch.Tensor): A tensor of shape [K, d_out, d_in].
        m_phi_minus (torch.Tensor): A tensor of shape [K, d_out, d_in].
        k_y (int): Number of time steps to shift.

    Returns:
        torch.Tensor: The spectral component tensor of shape [bsz, sl, d_out].
    """
    sigma, phi = eigh
    _, K = phi.shape

    # Compute U⁺ and U⁻
    U_plus, U_minus = conv(inputs, phi)  # -> tuple of [bsz, sl, K, d_in]

    # Shift U⁺ and U⁻ k_y time steps
    if M_y is not None:
        k_y = M_y.shape[0]
        U_plus, U_minus = shift(U_plus, k_y), shift(U_minus, k_y)

    # Perform spectral filter on U⁺ and U⁻ w/ sigma
    sigma_root = (sigma**0.25).view(1, 1, K, 1)
    U_plus_filtered, U_minus_filtered = U_plus * sigma_root, U_minus * sigma_root

    # Sum M^{\phi +}_k \cdot U_plus_filtered across K filters
    spectral_plus = torch.einsum("bsKd,Kod->bso", U_plus_filtered, M_phi_plus)

    # Sum M^{\phi -}_k \cdot U_minus_filtered across K filters
    spectral_minus = torch.einsum("bsKd,Kod->bso", U_minus_filtered, M_phi_minus)

    return spectral_plus + spectral_minus
