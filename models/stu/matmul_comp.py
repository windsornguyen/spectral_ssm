import torch
import torch.nn.functional as F
import numpy as np


def benchmark_matmul(bsz, x, y, z, num_runs=1000):
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(bsz, x, y, device=device)
    b = torch.randn(bsz, y, z, device=device)

    methods = {
        "bmm": lambda: torch.bmm(a, b),
        "@": lambda: a @ b,
        "matmul": lambda: torch.matmul(a, b),
        "nn.functional.linear": lambda: torch.stack(
            [F.linear(a[i], b[i].t()) for i in range(bsz)]
        ),
        "einsum": lambda: torch.einsum("bij,bjk->bik", a, b),
        # tensordot is very slow
        # "tensordot": lambda: torch.tensordot(a, b, dims=([-1], [-2])),
    }

    results = {name: [] for name in methods}

    # Warm-up run
    for _ in range(1):
        for method in methods.values():
            method()
    torch.cuda.synchronize()

    for _ in range(num_runs):
        for name, method in methods.items():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            method()
            end_event.record()

            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)
            results[name].append(elapsed_time)

    # Calculate mean and standard deviation
    stats = {}
    for name, times in results.items():
        mean_time = np.mean(times)
        std_time = np.std(times)
        stats[name] = (mean_time, std_time)

    # Sort by mean time
    sorted_stats = sorted(stats.items(), key=lambda x: x[1][0])

    return sorted_stats


# Production-size inputs
bsz, x, y, z = 32, 1024, 1024, 1024

print(f"Benchmarking with tensor sizes: ({bsz}, {x}, {y}) @ ({bsz}, {y}, {z})")
results = benchmark_matmul(bsz, x, y, z)

print("\nResults (sorted by speed):")
print(f"{'Method':<25} {'Mean Time (ms)':<15} {'Std Dev (ms)':<15}")
print("-" * 55)
for name, (mean_time, std_time) in results:
    print(f"{name:<25} {mean_time:<15.4f} {std_time:<15.4f}")

# Calculate speedup relative to slowest method
slowest_time = results[-1][1][0]
print("\nSpeedup relative to slowest method:")
for name, (mean_time, _) in results:
    speedup = slowest_time / mean_time
    print(f"{name:<25} {speedup:.2f}x")

print(
    "\nNote: nn.functional.linear is implemented with a loop over the batch dimension, which may affect its performance."
)
