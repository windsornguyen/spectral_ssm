import torch
import time


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k == 0:
        return u
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted


def compute_ar(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    bsz, sl, d_out = y.shape
    k_u = m_u.shape[0]
    ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    for t in range(sl):
        for i in range(1, min(t + 1, k_y) + 1):
            y_shifted = shift(y, i)
            ar_component[:, t, :] += torch.einsum(
                "bd,od->bo", y_shifted[:, t, :], m_y[i - 1]
            )

        for i in range(1, min(t + 2, k_u + 1)):
            u_shifted = shift(u, i - 1)
            ar_component[:, t, :] += torch.einsum(
                "bd,di->bi", u_shifted[:, t, :], m_u[i - 1]
            )

    return ar_component



def compute_ar_opt(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    bsz, sl, d_out = y.shape
    _, _, d_in = u.shape
    k_u = m_u.shape[0]

    ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    max_k = max(k_y, k_u)
    for i in range(max_k):
        if i < k_y:
            y_shifted = shift(y, i + 1)
            ar_component += torch.einsum("bsd,od->bso", y_shifted, m_y[i])

        if i < k_u:
            u_shifted = shift(u, i)
            ar_component += torch.einsum("bsd,di->bsi", u_shifted, m_u[i])

    return ar_component


def compute_y_t(M_y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bsz, sl, d_out = y.shape
    k_y = M_y.shape[0]
    y_t = torch.zeros_like(y)

    # Add the k_y (filter) dimension and expand out the tensor
    expanded_y = y.unsqueeze(1).expand(bsz, k_y, sl, d_out)

    # Fuse the bsz and k_y dimensions in preparation for the bmm
    fused_y = expanded_y.reshape(bsz * k_y, sl, d_out)

    # bmm requirement: (b, n, M) x (b, M, p) -> (b, n, p)
    batched_M_y = M_y.transpose(-1, -2).repeat(bsz, 1, 1)  # (b, M, n)

    # Perform the bmm!
    o = torch.bmm(fused_y, batched_M_y)
    M_y_ = o.view(bsz, k_y, sl, d_out)

    # Sum over the results of each M_y filter
    for k in range(k_y):
        y_t[:, k + 1 :] += M_y_[:, k, : sl - k - 1]

    return y_t


def compute_u_t(M_u: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    bsz, sl, d_in = u.shape
    k_u = M_u.shape[0]

    # Add the k_u (filter) dimension and expand out the tensor
    expanded_u = u.unsqueeze(1).expand(bsz, k_u, sl, d_in)

    # Fuse the bsz and k_y dimensions in preparation for the bmm
    fused_u = expanded_u.reshape(bsz * k_u, sl, d_in)

    # Ensure M_u is properly batched
    batched_M_u = M_u.repeat(bsz, 1, 1)

    # Perform the bmm!
    o = torch.bmm(fused_u, batched_M_u)
    M_u_ = o.view(bsz, k_u, sl, -1)
    u_t = torch.zeros(bsz, sl, o.shape[-1], device=u.device)

    # Sum over the results of each M_u filter
    for k in range(k_u):
        u_t[:, k:] += M_u_[:, k, : sl - k]

    return u_t


def compute_ar_opt_opt(
    y: torch.Tensor, u: torch.Tensor, M_y: torch.Tensor, M_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    yt = compute_y_t(M_y, y)
    ut = compute_u_t(M_u, u)
    return yt + ut


def compute_y_t_shift(M_y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bsz, sl, d_out = y.shape
    k_y = M_y.shape[0]

    shifted_y = torch.stack([shift(y, k) for k in range(1, k_y + 1)], dim=1)
    reshaped_y = shifted_y.view(bsz * k_y, sl, d_out)
    repeated_M_y = M_y.transpose(-1, -2).repeat(bsz, 1, 1)

    o = torch.bmm(reshaped_y, repeated_M_y)
    o = o.view(bsz, k_y, sl, d_out)
    yt = torch.sum(o, dim=1)

    return yt


def compute_u_t_shift(M_u: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    bsz, sl, d_in = u.shape
    k_u = M_u.shape[0]

    shifted_u = torch.stack([shift(u, k) for k in range(k_u)], dim=1)
    reshaped_u = shifted_u.view(bsz * k_u, sl, d_in)
    repeated_M_u = M_u.repeat(bsz, 1, 1)

    o = torch.bmm(reshaped_u, repeated_M_u)
    o = o.view(bsz, k_u, sl, -1)
    ut = torch.sum(o, dim=1)

    return ut


def compute_ar_opt_opt_shift(
    y: torch.Tensor, u: torch.Tensor, M_y: torch.Tensor, M_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    yt = compute_y_t_shift(M_y, y)
    ut = compute_u_t_shift(M_u, u)
    return yt + ut


def compute_ar_opt_opt_combined(
    y: torch.Tensor, u: torch.Tensor, M_y: torch.Tensor, M_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    bsz, sl, d_out = y.shape
    _, _, d_in = u.shape
    k_y, k_u = M_y.shape[0], M_u.shape[0]

    expanded_y = y.unsqueeze(1).expand(bsz, k_y, sl, d_out)
    reshaped_y = expanded_y.reshape(bsz * k_y, sl, d_out)
    repeated_M_y = M_y.transpose(-1, -2).repeat(bsz, 1, 1)

    expanded_u = u.unsqueeze(1).expand(bsz, k_u, sl, d_in)
    reshaped_u = expanded_u.reshape(bsz * k_u, sl, d_in)
    repeated_M_u = M_u.repeat(bsz, 1, 1)

    o_y = torch.bmm(reshaped_y, repeated_M_y).view(bsz, k_y, sl, d_out)
    o_u = torch.bmm(reshaped_u, repeated_M_u).view(bsz, k_u, sl, -1)

    yt = torch.zeros_like(y)
    ut = torch.zeros(bsz, sl, o_u.shape[-1], device=u.device)

    for k in range(max(k_y, k_u)):
        if k < k_y:
            yt[:, k + 1 :] += o_y[:, k, : sl - k - 1]
        if k < k_u:
            ut[:, k:] += o_u[:, k, : sl - k]

    return yt + ut


def get_toeplitz(x: torch.Tensor, k: int, lower: bool = True) -> torch.Tensor:
    """
    Efficiently construct Toeplitz matrices for each batch and feature, up to k steps.

    Args:
    x (torch.Tensor): Input tensor of shape (bsz, sl, d)
    k (int): Number of steps to include
    lower (bool): If True, construct lower triangular Toeplitz matrices

    Returns:
    torch.Tensor: Toeplitz matrices of shape (bsz, sl, k, d)
    """
    bsz, sl, d = x.shape
    row_indices = torch.arange(sl, device=x.device)
    col_indices = torch.arange(k, device=x.device)
    indices = col_indices - row_indices.unsqueeze(1)

    if lower:
        mask = indices.le(0).unsqueeze(0).unsqueeze(-1)
    else:
        mask = indices.ge(0).unsqueeze(0).unsqueeze(-1)

    x_expanded = x.unsqueeze(2).expand(bsz, sl, k, d)
    shifted = x_expanded.gather(
        1, (-indices).clamp(min=0).unsqueeze(0).unsqueeze(-1).expand(bsz, sl, k, d)
    )
    result = shifted * mask.to(x.dtype)
    return result


def compute_ar_bmm(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    """
    Compute the AR component efficiently using batched operations.

    Args:
    y (torch.Tensor): Output tensor of shape (bsz, sl, d_out)
    u (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
    m_y (torch.Tensor): Output weight matrix of shape (k_y, d_out, d_out)
    m_u (torch.Tensor): Input weight matrix of shape (k_u, d_out, d_in)
    k_y (int): Number of steps for y

    Returns:
    torch.Tensor: AR component of shape (bsz, sl, d_out)
    """
    bsz, sl, d_out = y.shape
    k_u = m_u.size(0)

    y_toeplitz = get_toeplitz(y, k_y, lower=True)  # Lower triangular for y
    u_toeplitz = get_toeplitz(u, k_u, lower=False)  # Upper triangular for u

    y_toeplitz_t = y_toeplitz.transpose(1, 2)
    u_toeplitz_t = u_toeplitz.transpose(1, 2)

    ar_y = torch.einsum("bksd,kod->bsd", y_toeplitz_t, m_y)
    ar_u = torch.einsum("bksd,kod->bsd", u_toeplitz_t, m_u)

    # Apply shifting
    ar_y_shifted = torch.zeros_like(y)
    ar_y_shifted[:, 1:] = ar_y[:, :-1]

    return ar_y_shifted + ar_u


def run_tests():
    # Define test inputs
    bsz, sl, d_in, d_out, k_y = 2, 5, 2, 2, 2
    M_y = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    M_u = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    y = torch.tensor(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
            [[1, 0], [0, 1], [1, 1], [0, 0], [1, 1]],
        ],
        dtype=torch.float32,
    )
    u = torch.tensor(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
            [[1, 0], [0, 1], [1, 1], [0, 0], [1, 1]],
        ],
        dtype=torch.float32,
    )

    # Compute outputs using all functions
    functions = [
        compute_ar,
        compute_ar_opt,
        compute_ar_opt_opt,
        compute_ar_opt_opt_shift,
        compute_ar_opt_opt_combined,
        compute_ar_bmm,
    ]

    results = {}
    times = {}

    for func in functions:
        start_time = time.time()
        result = func(y, u, M_y, M_u, k_y)
        end_time = time.time()
        results[func.__name__] = result
        times[func.__name__] = end_time - start_time

    # Compare results
    base_result = results["compute_ar"]
    for func_name, result in results.items():
        if func_name != "compute_ar":
            is_close = torch.allclose(base_result, result, atol=1e-6)
            print(f"{func_name} matches compute_ar: {is_close}")
            print(f"Execution time: {times[func_name]:.6f} seconds")

    print("\nDetailed results:")
    for func_name, result in results.items():
        print(f"\n{func_name}:")
        print(result)


if __name__ == "__main__":
    run_tests()
