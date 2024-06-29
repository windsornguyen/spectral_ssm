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
    bsz, sl, d = u.shape
    padding = torch.zeros(bsz, k, d, device=u.device)
    if k < sl:
        shifted = torch.cat([padding, u[:, :-k]], dim=1)
    else:
        shifted = padding[:, :sl]
    return shifted


def compute_ar(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    bsz, sl, d_out = y.shape
    ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    for t in range(sl):
        for i in range(1, min(t + 1, k_y) + 1):
            y_shifted = shift(y, i)
            ar_component[:, t, :] += torch.einsum(
                "bd,od->bo", y_shifted[:, t, :], m_y[i - 1]
            )

        for i in range(min(t + 1, k_y + 1)):
            u_shifted = shift(u, i)
            ar_component[:, t, :] += torch.einsum(
                "bd,di->bi", u_shifted[:, t, :], m_u[i]
            )

    return ar_component


def compute_ar_opt(
    y: torch.Tensor, u: torch.Tensor, m_y: torch.Tensor, m_u: torch.Tensor, k_y: int
) -> torch.Tensor:
    bsz, sl, d_out = y.shape
    k_u = m_u.shape[0]
    ar_component = torch.zeros(bsz, sl, d_out, device=y.device)

    for i in range(k_y):
        y_shifted = shift(y, i + 1)
        ar_component += torch.einsum("btd,od->bto", y_shifted, m_y[i])

    for i in range(k_u):
        u_shifted = shift(u, i)
        ar_component += torch.einsum("btd,di->bti", u_shifted, m_u[i])

    return ar_component


def compute_y_t(M_y, y):
    bsz, sl, d_out = y.shape
    k_y = M_y.shape[0]
    expanded_y = y.unsqueeze(1).expand(bsz, k_y, sl, d_out)
    reshaped_y = expanded_y.reshape(bsz * k_y, sl, d_out)
    repeated_M_y = M_y.transpose(-1, -2).repeat(bsz, 1, 1) # TODO: Why is this transpose necessary?

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
    for k in range(k_u):
        ut[:, k:] += o[:, k, : sl - k]
    return ut


def compute_ar_opt_opt(y, u, M_y, M_u, k_y):
    yt = compute_y_t(M_y, y)
    ut = compute_u_t(M_u, u)
    return yt + ut

def compute_ar_opt_opt_combined(y, u, M_y, M_u, k_y):
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
            yt[:, k + 1:] += o_y[:, k, :sl - k - 1]
        if k < k_u:
            ut[:, k:] += o_u[:, k, :sl - k]
    
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


def compute_ar_opt_opt_shift(y, u, M_y, M_u, k_y):
    yt = compute_y_t_shift(M_y, y)
    ut = compute_u_t_shift(M_u, u)
    return yt + ut


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
        # "compute_ar_opt_shift": compute_ar_opt_opt_shift,
        "compute_ar_opt_opt_slice": compute_ar_opt_opt,
        "compute_ar_opt_opt_slice_combined": compute_ar_opt_opt_combined,
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
        {"bsz": 8, "sl": 1000, "d_in": 512, "d_out": 512, "k_y": 2},
        {"bsz": 8, "sl": 1000, "d_in": 512, "d_out": 512, "k_y": 2},
    ]

    for config in test_configs:
        print(f"\nRunning test with configuration: {config}")
        timing_results, equality_results = run_tests(**config)
        table = format_results(timing_results, equality_results, config)
        print(table)


if __name__ == "__main__":
    main()
