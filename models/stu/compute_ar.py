import torch
from prettytable import PrettyTable

# Check for CUDA device and set as default if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Setting the seed for reproducibility
torch.manual_seed(1337)


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k == 0:
        return u
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted


def compute_ar(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    # bsz, sl, d_out = y.shape
    # ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    # for t in range(sl):
    #     for i in range(1, min(t + 1, k_y) + 1):
    #         y_shifted = shift(y, i)
    #         ar_component[:, t, :] += torch.einsum(
    #             "bd,od->bo", y_shifted[:, t, :], m_y[i - 1]
    #         )

    #     for i in range(min(t + 1, k_y + 1)):
    #         u_shifted = shift(u, i)
    #         ar_component[:, t, :] += torch.einsum(
    #             "bd,di->bi", u_shifted[:, t, :], m_u[i]
    #         )

    # return ar_component
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
    """
    Computes the Mʸ-summation within the autoregressive
    component of the AR-STU model using past outputs.

    Args:
        M_y (torch.Tensor): Output weight matrices of shape (k_y, d_out, d_out)
        y (torch.Tensor): Output tensor of shape (bsz, sl, d_out)

    Returns:
        torch.Tensor: Autoregressive component of shape (bsz, sl, d_out)

    Note:
        k_y: Number of past outputs to consider
        bsz: Batch size
        sl: Sequence length
        d_out: Output dimension
    """
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
    """
    Computes the Mᵘ-summation within the autoregressive
    component of the AR-STU model using past inputs.

    Args:
        M_u (torch.Tensor): Input weight matrices of shape (k_u, d_in, d_out)
        u (torch.Tensor): Input tensor of shape (bsz, sl, d_in)

    Returns:
        torch.Tensor: Input component of shape (bsz, sl, d_out)

    Note:
        k_u: Number of past inputs to consider
        bsz: Batch size
        sl: Sequence length
        d_in: Input dimension
        d_out: Output dimension
    """
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
    # bsz, sl, d_in = u.shape
    # k_u = M_u.shape[0]
    # u_t = torch.zeros(bsz, sl, M_u.shape[2], device=u.device)

    # # Compute the contribution from past inputs
    # for j in range(1, k_u + 1):
    #     if sl - j >= 0:
    #         u_t[:, j - 1 :] += torch.bmm(u[:, : sl - j + 1], M_u[j - 1].expand(bsz, -1, -1))

    # return u_t


def compute_ar_bmm_sliced(
    y: torch.Tensor, u: torch.Tensor, M_y: torch.Tensor, M_u: torch.Tensor, k_y: int
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
    yt = compute_y_t(M_y, y)
    ut = compute_u_t(M_u, u)
    return yt + ut


def compute_ar_opt_bmm_sliced_1loop(y, u, M_y, M_u, k_y):
    bsz, sl, d_out = y.shape
    _, _, d_in = u.shape
    k_y, k_u = M_y.shape[0], M_u.shape[0]

    # Prepare y inputs
    expanded_y = y.unsqueeze(1).expand(bsz, k_y, sl, d_out)
    reshaped_y = expanded_y.reshape(bsz * k_y, sl, d_out)
    repeated_M_y = M_y.transpose(-1, -2).repeat(bsz, 1, 1)

    # Prepare u inputs
    expanded_u = u.unsqueeze(1).expand(bsz, k_u, sl, d_in)
    reshaped_u = expanded_u.reshape(bsz * k_u, sl, d_in)
    repeated_M_u = M_u.repeat(bsz, 1, 1)

    # Compute y_t and u_t in parallel
    o_y = torch.bmm(reshaped_y, repeated_M_y).view(bsz, k_y, sl, d_out)
    o_u = torch.bmm(reshaped_u, repeated_M_u).view(bsz, k_u, sl, -1)

    # Initialize output tensors
    yt = torch.zeros_like(y)
    ut = torch.zeros(bsz, sl, o_u.shape[-1], device=u.device)

    # Fill output tensors
    for k in range(max(k_y, k_u)):
        if k < k_y:
            yt[:, k + 1 :] += o_y[:, k, : sl - k - 1]
        if k < k_u:
            ut[:, k:] += o_u[:, k, : sl - k]

    return yt + ut


def compute_y_t_shift(M_y, y):
    bsz, sl, d_out = y.shape
    k_y = M_y.shape[0]

    shifted_y = torch.stack([shift(y, k) for k in range(1, k_y + 1)], dim=1)
    reshaped_y = shifted_y.view(bsz * k_y, sl, d_out)
    repeated_M_y = M_y.transpose(-1, -2).repeat(bsz, 1, 1)

    o = torch.bmm(reshaped_y, repeated_M_y)
    o = o.view(bsz, k_y, sl, d_out)
    yt = torch.sum(o, dim=1)

    return yt


def compute_u_t_shift(M_u, u):
    bsz, sl, d_in = u.shape
    k_u = M_u.shape[0]

    shifted_u = torch.stack([shift(u, k) for k in range(k_u)], dim=1)
    reshaped_u = shifted_u.view(bsz * k_u, sl, d_in)
    repeated_M_u = M_u.repeat(bsz, 1, 1)

    o = torch.bmm(reshaped_u, repeated_M_u)
    o = o.view(bsz, k_u, sl, -1)
    ut = torch.sum(o, dim=1)

    return ut


def compute_ar_opt_bmm_shift(y, u, M_y, M_u, k_y):
    yt = compute_y_t_shift(M_y, y)
    ut = compute_u_t_shift(M_u, u)
    return yt + ut


def compute_ar_opt_einsum_shift(y, u, M_y, M_u, k_y):
    bsz, sl, d_out = y.shape
    _, _, d_in = u.shape
    k_u = M_u.shape[0]

    # Create tensors of shifted y and u
    y_shifts = torch.stack([shift(y, i + 1) for i in range(k_y)], dim=1)
    u_shifts = torch.stack([shift(u, i) for i in range(k_u)], dim=1)

    # Compute AR components using einsum
    
    ar_y = torch.einsum("bksd,kod->bso", y_shifts, M_y)
    ar_u = torch.einsum("bksd,kdi->bsi", u_shifts, M_u)

    return ar_y + ar_u


def get_toeplitz(x: torch.Tensor, k: int, lower: bool = True) -> torch.Tensor:
    """
    Efficiently construct Toeplitz matrices for each batch and feature, up to k steps.

    Args:
    y (torch.Tensor): Input tensor of shape (bsz, sl, d)
    k (int): Number of steps to include

    Returns:
    torch.Tensor: Upper triangular Toeplitz matrices of shape (bsz, sl, k, d)
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


# !! NOTE: Very memory-intensive implementation. Also gives incorrect output !!
def compute_ar_bmm(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y
) -> torch.Tensor:
    """
    Compute the AR component efficiently using batched operations.

    Args:
    y (torch.Tensor): Output tensor of shape (bsz, sl, d_out)
    u (torch.Tensor): Input tensor of shape (bsz, sl, d_in)
    m_y (torch.Tensor): Output weight matrix of shape (k_y, d_out, d_out)
    m_u (torch.Tensor): Input weight matrix of shape (k_u, d_out, d_in)

    Returns:
    torch.Tensor: AR component of shape (bsz, sl, d_out)
    """
    bsz, sl, d_out = y.shape
    _, _, d_in = u.shape
    k_u = m_u.size(0)

    # Create Toeplitz matrices
    y_toeplitz = get_toeplitz(y, k_y + 1, lower=True)[:, :, 1:]  # Start from 1 for y
    u_toeplitz = get_toeplitz(u, k_u, lower=False)

    # Compute AR components
    ar_y = torch.einsum("bksd,kod->bsd", y_toeplitz.transpose(1, 2), m_y)
    ar_u = torch.einsum("bksd,kdi->bsi", u_toeplitz.transpose(1, 2), m_u)

    # Shift y component forward
    ar_y_shifted = torch.zeros_like(y)
    ar_y_shifted[:, 1:] = ar_y[:, :-1]

    return ar_y_shifted + ar_u


def generate_random_inputs(
    bsz: int, sl: int, d_in: int, d_out: int, k_y: int
) -> tuple[torch.Tensor, ...]:
    print(
        f"\nGenerating random inputs with bsz={bsz}, sl={sl}, d_in={d_in}, d_out={d_out}, k_y={k_y}"
    )
    m_y = torch.randn(k_y, d_out, d_out, device=device)
    m_u = torch.randn(k_y + 1, d_out, d_in, device=device)
    y = torch.randn(bsz, sl, d_out, device=device)
    u = torch.randn(bsz, sl, d_in, device=device)
    print("Generated input shapes:")
    print(f"m_y: {m_y.shape}")
    print(f"m_u: {m_u.shape}")
    print(f"y: {y.shape}")
    print(f"u: {u.shape}")
    return m_y, m_u, y, u


def time_function(func, *args):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up run
    _ = func(*args)
    torch.cuda.synchronize()

    # Timed run
    start_event.record()
    output = func(*args)
    end_event.record()

    torch.cuda.synchronize()

    return output, start_event.elapsed_time(end_event)


def compare_outputs(
    outputs: dict[str, torch.Tensor],
    reference_output: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, tuple[bool, float, float]]:
    results = {}
    for name, output in outputs.items():
        are_equal = torch.allclose(reference_output, output, rtol=rtol, atol=atol)
        if not are_equal:
            diff = torch.abs(reference_output - output)
            max_diff = torch.max(diff).item()
            relative_diff = torch.max(diff / torch.abs(reference_output)).item()
            results[name] = (are_equal, max_diff, relative_diff)
        else:
            results[name] = (are_equal, 0, 0)
    return results


def run_tests(
    bsz: int, sl: int, d_in: int, d_out: int, k_y: int, num_runs: int = 10
) -> tuple[dict[str, list[float]], dict[str, tuple[bool, float, float]]]:
    m_y, m_u, y, u = generate_random_inputs(bsz, sl, d_in, d_out, k_y)

    functions = {
        "compute_ar": compute_ar,
        "compute_ar_opt": compute_ar_opt,
        "compute_ar_bmm_sliced": compute_ar_bmm_sliced,
        "compute_ar_opt_bmm_sliced_1loop": compute_ar_opt_bmm_sliced_1loop,
        "compute_ar_opt_bmm_shift": compute_ar_opt_bmm_shift,
        "compute_ar_opt_einsum_shift": compute_ar_opt_einsum_shift,
        "compute_ar_bmm": compute_ar_bmm,
    }

    results = {name: [] for name in functions}
    outputs = {}

    for name, func in functions.items():
        for _ in range(num_runs):
            output, time_taken = time_function(func, y, u, m_y, m_u, k_y)
            results[name].append(time_taken)
        outputs[name] = output

    equality_results = compare_outputs(outputs, outputs["compute_ar"])

    return results, equality_results


def format_results(
    timing_results: dict[str, list[float]],
    equality_results: dict[str, tuple[bool, float, float]],
    config: dict[str, any],
) -> PrettyTable:
    table = PrettyTable()
    table.field_names = [
        "Function",
        "Avg Time (ms)",
        "Speedup",
        "Max Diff",
        "Max Rel Diff",
        "Equal",
    ]

    base_time = sum(timing_results["compute_ar"]) / len(timing_results["compute_ar"])

    for name, times in timing_results.items():
        avg_time = sum(times) / len(times)
        speedup = base_time / avg_time if name != "compute_ar" else 1.0
        equal, max_diff, rel_diff = equality_results.get(name, (True, 0, 0))

        table.add_row(
            [
                name,
                f"{avg_time:.4f}",
                f"{speedup:.2f}x",
                f"{max_diff:.6f}",
                f"{rel_diff:.6f}",
                "Yes" if equal else "No",
            ]
        )

    table.title = f"Results for config: {config}"
    return table


def main():
    test_configs = [
        {"bsz": 2, "sl": 4, "d_in": 4, "d_out": 4, "k_y": 2},
        {"bsz": 4, "sl": 500, "d_in": 256, "d_out": 256, "k_y": 2},
        {"bsz": 8, "sl": 1000, "d_in": 512, "d_out": 512, "k_y": 2},
        {"bsz": 8, "sl": 2048, "d_in": 1024, "d_out": 1024, "k_y": 2},
    ]

    for config in test_configs:
        print(f"\nRunning test with configuration: {config}")
        timing_results, equality_results = run_tests(**config)
        table = format_results(timing_results, equality_results, config)
        print(table)


if __name__ == "__main__":
    main()
