import torch
import numpy as np
import time
import numpy.linalg as npla
import torch.autograd.profiler as profiler

@torch.jit.script
def get_hankel_matrix(n: int, device: torch.device) -> torch.Tensor:
    """Generate a spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.

    Returns:
        torch.Tensor: A spectral Hankel matrix of shape [n, n].
    """
    indices = torch.arange(1, n + 1, device=device)
    sum_indices = indices[:, None] + indices[None, :]
    z = 2 / (sum_indices ** 3 - sum_indices)

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

@torch.jit.script
def get_top_hankel_eigh(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

    Args:
        n (int): Number of rows in square spectral Hankel matrix.
        k (int): Number of eigenvalues to return.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of eigenvalues of shape [k,] and 
            eigenvectors of shape [n, k].
    """
    eig_vals, eig_vecs = torch.linalg.eigh(get_hankel_matrix(n, device=device))
    return eig_vals[-k:], eig_vecs[:, -k:]

def get_top_hankel_eigh_(
    n: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Get top k eigenvalues and eigenvectors of spectral Hankel matrix.

  Args:
    n: Number of rows in square spectral Hankel matrix.
    k: Number of eigenvalues to return.

  Returns:
    A tuple of eigenvalues of shape [k,] and eigenvectors of shape [l, k].
  """
  eig_vals, eig_vecs = np.linalg.eigh(get_hankel_matrix_(n))
  return eig_vals[-k:], eig_vecs[:, -k:]

# Set seed
np.random.seed(1337)

# Prepare random data for testing
n = np.random.randint(1, 1024)  # Random size for the matrix
k = min(n, np.random.randint(1, 50))  # Random number of eigenvalues/eigenvectors to extract
print(f'Testing top {k} eigenvalues and eigenvectors for a {n}x{n} Hankel matrix.')

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Warm up the JIT compiler
_ = get_top_hankel_eigh(n, k, device)

# Profile PyTorch
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    _ = get_top_hankel_eigh(n, k, device)

# Print the profiling results for PyTorch
print(prof.key_averages().table(sort_by="cpu_time_total"))

# Benchmark and test outputs
start_time_torch = time.time()
eig_vals_torch, eig_vecs_torch = get_top_hankel_eigh(n, k, device)
time_torch = time.time() - start_time_torch

start_time_numpy = time.time()
eig_vals_numpy, eig_vecs_numpy = get_top_hankel_eigh_(n, k)
time_numpy = time.time() - start_time_numpy

# Move PyTorch tensors back to CPU
eig_vals_torch = eig_vals_torch.cpu()
eig_vecs_torch = eig_vecs_torch.cpu()

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (NumPy): {time_numpy:.6f}s')

# Sorting the eigenvalues and eigenvectors after they are returned, in descending order by eigenvalue magnitude
sorted_indices_torch = eig_vals_torch.abs().sort(descending=True).indices
eig_vals_torch = eig_vals_torch[sorted_indices_torch]
eig_vecs_torch = eig_vecs_torch[:, sorted_indices_torch]

sorted_indices_numpy = np.argsort(-np.abs(eig_vals_numpy))
eig_vals_numpy = eig_vals_numpy[sorted_indices_numpy]
eig_vecs_numpy = eig_vecs_numpy[:, sorted_indices_numpy]

# Compare the top k eigenvalues and eigenvectors
print('Comparing the top k eigenvalues and eigenvectors...')

# Check if the eigenvalues are close enough
if np.allclose(eig_vals_torch.numpy()[:k], eig_vals_numpy[:k], atol=1e-8):
    print('Top k eigenvalues are close enough between PyTorch and NumPy.')
else:
    print('Top k eigenvalues differ more than acceptable tolerance between PyTorch and NumPy.')

# Check if the eigenvectors are close enough
if np.allclose(eig_vecs_torch.numpy()[:, :k], eig_vecs_numpy[:, :k], atol=1e-8):
    print('Top k eigenvectors are close enough between PyTorch and NumPy.')
else:
    print('Top k eigenvectors differ more than acceptable tolerance between PyTorch and NumPy.')

# Calculate and print the condition number
cond_number_torch = npla.cond(eig_vecs_torch.numpy())
print(f'\nCondition number of the Torch matrix: {cond_number_torch}')
cond_number_numpy = npla.cond(eig_vecs_numpy)
print(f'Condition number of the NumPy matrix: {cond_number_numpy}')
