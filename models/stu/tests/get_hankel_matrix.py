import torch
import numpy as np
import time
import torch.autograd.profiler as profiler

@torch.jit.script
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

def get_hankel_matrix_(
    n: int,
) -> np.ndarray:
    """Generate a spectral Hankel matrix.

    Args:
        n: Number of rows in square spectral Hankel matrix.

    Returns:
        A spectral Hankel matrix of shape [n, n].
    """
    z = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            z[i - 1, j - 1] = 2 / ((i + j) ** 3 - (i + j))
    return z

# Set seed
np.random.seed(0)

# Prepare random data for testing
n = np.random.randint(1, 1024)  # Random size for the matrix
print(f'Testing matrix size: {n}x{n}')

# Warm-up JIT compilation
_ = get_hankel_matrix(n)
_ = get_hankel_matrix_(n)

# Profile PyTorch
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    hankel_torch = get_hankel_matrix(n)

# Print the profiling results for PyTorch
print(prof.key_averages().table(sort_by="cpu_time_total"))

# Benchmark and test outputs
start_time_torch = time.time()
hankel_torch = get_hankel_matrix(n).numpy()
time_torch = time.time() - start_time_torch

start_time_numpy = time.time()
hankel_numpy = get_hankel_matrix_(n)
time_numpy = time.time() - start_time_numpy

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (NumPy): {time_numpy:.6f}s')

# Check outputs and compare
if not np.allclose(hankel_torch, hankel_numpy, atol=1e-8):
    print('Values differ more than acceptable tolerance.')
    difference_matrix = np.abs(hankel_torch - hankel_numpy)
    print('Differing Values (showing first few):')
    count_diffs = 0
    for i in range(n):
        for j in range(n):
            if difference_matrix[i][j] > 1e-8:
                print(f'Index: ({i}, {j}), Torch: {hankel_torch[i][j]}, NumPy: {hankel_numpy[i][j]}, Diff: {difference_matrix[i][j]}')
                count_diffs += 1
                if count_diffs > 10:  # Limit to first few differences
                    break
        if count_diffs > 10:
            break
else:
    print('Outputs are sufficiently close.')
