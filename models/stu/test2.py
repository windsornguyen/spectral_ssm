import torch
import time

# Check for CUDA device and set as default if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Setting the seed for reproducibility
torch.manual_seed(42)


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


def compute_ar_cache(  # misnomer but too lazy to change it
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


def initialize_test_data(bsz, sl, d_in, d_out, k_y):
    m_y = torch.randn(k_y, d_out, d_out, device=device)
    m_u = torch.randn(k_y + 1, d_out, d_in, device=device)
    y = torch.randn(bsz, sl, d_out, device=device)
    u = torch.randn(bsz, sl, d_in, device=device)
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


def compare_outputs(output_ar, output_ar_cache, rtol=1e-5, atol=1e-5):
    are_equal = torch.allclose(output_ar, output_ar_cache, rtol=rtol, atol=atol)
    if not are_equal:
        diff = torch.abs(output_ar - output_ar_cache)
        not_close = ~torch.isclose(output_ar, output_ar_cache, rtol=rtol, atol=atol)
        indices = torch.nonzero(not_close, as_tuple=False)

        print("Differences found. First 5 differences:")
        for idx in indices[:5]:
            b, t, d = idx
            print(f"  Index [batch={b}, time={t}, dim={d}]:")
            print(f"    compute_ar value:       {output_ar[b, t, d]:.6f}")
            print(f"    compute_ar_cache value: {output_ar_cache[b, t, d]:.6f}")
            print(f"    Absolute difference:    {diff[b, t, d]:.6f}")
            print(
                f"    Relative difference:    {diff[b, t, d] / torch.abs(output_ar[b, t, d]):.6f}"
            )
            print()

    return are_equal


def run_tests(bsz, sl, d_in, d_out, k_y, num_runs=10):
    m_y, m_u, y, u = initialize_test_data(bsz, sl, d_in, d_out, k_y)

    compute_ar_times = []
    compute_ar_cache_times = []

    # Check equality
    output_ar = compute_ar(y, u, m_y, m_u, k_y)
    output_ar_cache = compute_ar_cache(y, u, m_y, m_u, k_y)

    are_equal = compare_outputs(output_ar, output_ar_cache)

    for _ in range(num_runs):
        _, time_ar = time_function(compute_ar, y, u, m_y, m_u, k_y)
        compute_ar_times.append(time_ar)

        _, time_ar_cache = time_function(compute_ar_cache, y, u, m_y, m_u, k_y)
        compute_ar_cache_times.append(time_ar_cache)

    return compute_ar_times, compute_ar_cache_times, are_equal


def print_results(ar_times, ar_cache_times, are_equal):
    print(f"compute_ar average time: {sum(ar_times) / len(ar_times):.4f} ms")
    print(
        f"compute_ar_cache average time: {sum(ar_cache_times) / len(ar_cache_times):.4f} ms"
    )

    if sum(ar_times) < sum(ar_cache_times):
        print("compute_ar is faster")
    else:
        print("compute_ar_cache is faster")

    print(f"Outputs are equal: {are_equal}")


def main():
    test_configs = [
        {"bsz": 32, "sl": 100, "d_in": 64, "d_out": 64, "k_y": 5},
        {"bsz": 64, "sl": 200, "d_in": 128, "d_out": 128, "k_y": 10},
        # {"bsz": 128, "sl": 500, "d_in": 256, "d_out": 256, "k_y": 20},
    ]

    for config in test_configs:
        print(f"\nRunning test with configuration: {config}")
        ar_times, ar_cache_times, are_equal = run_tests(**config)
        print_results(ar_times, ar_cache_times, are_equal)


if __name__ == "__main__":
    main()
