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
    bsz, sl, K, d = u.shape
    padding = torch.zeros(bsz, k, K, d, device=u.device)
    if k < sl:
        shifted = torch.cat([padding, u[:, :-k]], dim=1)
    else:
        shifted = padding[:, :sl]
    return shifted


def compute_spectral(
    U_plus: torch.Tensor,
    U_minus: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    sigma: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    K, d_out, d_in = m_phi_plus.shape
    bsz, sl, _, _ = U_plus.shape

    # Shift U_plus and U_minus by k_y time steps
    U_plus_shifted = shift(U_plus, k_y)
    U_minus_shifted = shift(U_minus, k_y)

    # Compute σ^(1/4)
    sigma_root = (sigma**0.25).view(1, 1, K, 1)

    # Compute the spectral component
    U_plus_weighted = U_plus_shifted * sigma_root

    spectral_plus = torch.einsum("bski,kio->bso", U_plus_weighted, m_phi_plus)

    U_minus_weighted = U_minus_shifted * sigma_root
    spectral_minus = torch.einsum("bski,kio->bso", U_minus_weighted, m_phi_minus)

    result = spectral_plus + spectral_minus

    return result


def compute_spectral_opt(
    U_plus: torch.Tensor,
    U_minus: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    sigma: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    # Shift U_plus and U_minus by k_y time steps
    U_shifted_plus = shift(U_plus, k_y)
    U_shifted_minus = shift(U_minus, k_y)

    # Compute σ^(1/4)
    sigma_root = (sigma**0.25).view(1, 1, -1, 1)

    # Weight U_shifted with sigma_root
    U_weighted_plus = U_shifted_plus * sigma_root
    U_weighted_minus = U_shifted_minus * sigma_root

    
    bsz, sl, K, d_in = U_weighted_plus.shape
    d_out = m_phi_plus.shape[-1]

    # Add bsz dim to m_phi
    m_phi_plus = m_phi_plus.unsqueeze(0).expand(bsz, -1, d_out, d_out)
    m_phi_minus = m_phi_minus.unsqueeze(0).expand(bsz, -1, d_out, d_out)
    
    m_phi_plus = m_phi_plus.reshape(bsz * K, d_out, d_out)
    m_phi_minus = m_phi_minus.reshape(bsz * K, d_out, d_out)

    U_weighted_plus = U_weighted_plus.permute(0, 2, 1, 3).reshape(bsz * K, sl, d_in)
    U_weighted_minus = U_weighted_minus.permute(0, 2, 1, 3).reshape(bsz * K, sl, d_in)

    
    spectral_plus = torch.bmm(U_weighted_plus, m_phi_plus)
    spectral_minus = torch.bmm(U_weighted_minus, m_phi_minus)

    spectral_plus = spectral_plus.view(bsz, sl, K, d_out)
    spectral_minus = spectral_plus.view(bsz, sl, K, d_out)
    return spectral_plus + spectral_minus


def generate_random_inputs(
    bsz: int, sl: int, d_in: int, d_out: int, K: int
) -> tuple[torch.Tensor, ...]:
    print(
        f"\nGenerating random inputs with bsz={bsz}, sl={sl}, d_in={d_in}, d_out={d_out}, K={K}"
    )
    U_plus = torch.randn(bsz, sl, K, d_in, device=device)
    U_minus = torch.randn(bsz, sl, K, d_in, device=device)
    m_phi_plus = torch.randn(K, d_out, d_in, device=device)
    m_phi_minus = torch.randn(K, d_out, d_in, device=device)
    sigma = torch.rand(K, device=device)
    print("Generated input shapes:")
    print(f"U_plus: {U_plus.shape}")
    print(f"U_minus: {U_minus.shape}")
    print(f"m_phi_plus: {m_phi_plus.shape}")
    print(f"m_phi_minus: {m_phi_minus.shape}")
    print(f"sigma: {sigma.shape}")
    return U_plus, U_minus, m_phi_plus, m_phi_minus, sigma


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
    rtol: float = 1e-3,
    atol: float = 1e-3,
    eps: float = 1e-8,
) -> dict[str, tuple[bool, float, float]]:
    results = {}
    for name, output in outputs.items():
        are_equal = torch.allclose(reference_output, output, rtol=rtol, atol=atol)
        if not are_equal:
            diff = torch.abs(reference_output - output)
            max_diff = torch.max(diff).item()
            relative_diff = torch.max(diff / torch.abs(reference_output) + eps).item()
            results[name] = (are_equal, max_diff, relative_diff)
        else:
            results[name] = (are_equal, 0, 0)
    return results


def run_tests(
    bsz: int, sl: int, d_in: int, d_out: int, K: int, k_y: int, num_runs: int = 30
) -> tuple[dict[str, list[float]], dict[str, tuple[bool, float, float]]]:
    U_plus, U_minus, m_phi_plus, m_phi_minus, sigma = generate_random_inputs(
        bsz, sl, d_in, d_out, K
    )

    functions = {
        "compute_spectral": compute_spectral,
        "compute_spectral_opt": compute_spectral_opt,
        # "compute_spectral_corrected": compute_spectral_corrected,
    }

    results = {name: [] for name in functions}
    outputs = {}

    print("\nStarting warmup...")
    for func in functions.values():
        for _ in range(5):
            _ = func(U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y)

    print("\nStarting timed runs...")
    for name, func in functions.items():
        for _ in range(num_runs):
            output, time_taken = time_function(
                func, U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y
            )
            results[name].append(time_taken)
        outputs[name] = output

    equality_results = compare_outputs(outputs, outputs["compute_spectral"])

    return results, equality_results


def format_results(
    timing_results: dict[str, list[float]],
    equality_results: dict[str, tuple[bool, float, float]],
    config: dict[str, any],
    eps: float = 1e-8,
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

    base_time = sum(timing_results["compute_spectral"]) / (
        len(timing_results["compute_spectral"]) + eps
    )

    for name, times in timing_results.items():
        avg_time = sum(times) / (len(times) + eps)
        speedup = base_time / (avg_time + eps) if name != "compute_spectral" else 1.0
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
        {"bsz": 8, "sl": 1000, "d_in": 512, "d_out": 512, "K": 24, "k_y": 2},
        {"bsz": 8, "sl": 1000, "d_in": 512, "d_out": 512, "K": 24, "k_y": 2},
    ]

    for config in test_configs:
        print(f"\nRunning test with configuration: {config}")
        timing_results, equality_results = run_tests(**config)
        table = format_results(timing_results, equality_results, config)
        print(table)


if __name__ == "__main__":
    main()
