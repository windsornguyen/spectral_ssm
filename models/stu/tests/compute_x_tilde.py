import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
from shift import shift_torch, shift_jax
from conv import conv_torch, conv_jax
import torch.autograd.profiler as profiler


def compute_x_tilde_torch(
    inputs: torch.Tensor, eigh: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Compute the x_tilde component of spectral state space model.

    Args:
        inputs (torch.Tensor): A tensor of shape [bsz, sl, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): A tuple of eigenvalues of shape [k,] and
            eigenvectors of shape [sl, k].

    Returns:
        torch.Tensor: x_tilde: A tensor of shape [bsz, sl, k * d_in].
    """
    eig_vals, eig_vecs = eigh
    k = eig_vals.size(0)
    bsz, sl, d_in = inputs.shape

    # Project inputs into the spectral basis
    x_spectral = conv_torch(eig_vecs, inputs) # -> [bsz, sl, k, d_in]

    # Reshape dims of eig_vals to match dims of x_spectral
    eig_vals = eig_vals.view(1, 1, k, 1) # -> [1, 1, k, 1]

    # Perform spectral filtering on x to obtain x_tilde
    x_tilde = x_spectral * eig_vals ** 0.25

    # TODO: May have to adjust this once we introduce autoregressive component.
    # Reshape x_tilde so that it's matmul-compatible with m_phi
    return x_tilde.view(bsz, sl, k * d_in)


@jax.jit
def compute_x_tilde_jax(
    inputs: jnp.ndarray, eigh: tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Compute the x_tilde component of spectral SSM using JAX.

    Args:
        inputs (jnp.ndarray): A tensor of shape [bsz, seq_len, d_in].
        eigh (tuple[jnp.ndarray, jnp.ndarray]): A tuple of eigenvalues of shape [k,] and
            eigenvectors of shape [seq_len, k].

    Returns:
        jnp.ndarray: x_tilde: A tensor of shape [bsz, seq_len, k * d_in].
    """
    eig_vals, eig_vecs = eigh
    bsz, l, _ = inputs.shape

    x_tilde = conv_jax(eig_vecs, inputs)
    x_tilde *= jnp.expand_dims(eig_vals**0.25, axis=(0, 2))
    # NOTE: Shifting twice is incorrect as of now, noted by Evan.
    # return shift_jax(shift_jax(x_tilde.reshape((l, -1))))
    return x_tilde.reshape((bsz, l, -1))


# Set a seed
np.random.seed(1337)

# Prepare random data for testing
batch_size = np.random.randint(1, 10)
seq_len = np.random.randint(4, 32)
d_out = np.random.randint(4, 32)
print(
    f'Testing compute_x_tilde function for inputs of shape [{batch_size}, {seq_len}, {d_out}] and eig_vals/eig_vecs of shape [{d_out}].'
)

# Create random input data
inputs_np = np.random.rand(batch_size, seq_len, d_out).astype(np.float32)
eig_vals_np = np.random.rand(d_out).astype(np.float32)
eig_vecs_np = np.random.rand(seq_len, d_out).astype(
    np.float32
)  # Changed shape to [seq_len, d_out]

inputs_torch = torch.from_numpy(inputs_np)
eig_vals_torch = torch.from_numpy(eig_vals_np)
eig_vecs_torch = torch.from_numpy(eig_vecs_np)

inputs_jax = jnp.array(inputs_np)
eig_vals_jax = jnp.array(eig_vals_np)
eig_vecs_jax = jnp.array(eig_vecs_np)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs_torch = inputs_torch.to(device)
eig_vals_torch = eig_vals_torch.to(device)
eig_vecs_torch = eig_vecs_torch.to(device)

# Warm up the JIT compilers
_ = compute_x_tilde_torch(inputs_torch, (eig_vals_torch, eig_vecs_torch))
_ = compute_x_tilde_jax(
    inputs_jax, (eig_vals_jax, eig_vecs_jax)
).block_until_ready()

# Profile PyTorch
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    result_torch = compute_x_tilde_torch(
        inputs_torch, (eig_vals_torch, eig_vecs_torch)
    )

# Print the profiling results for PyTorch
# print(prof.key_averages().table(sort_by='cpu_time_total'))

# Benchmark PyTorch
start_time_torch = time.time()
result_torch = compute_x_tilde_torch(
    inputs_torch, (eig_vals_torch, eig_vecs_torch)
)
time_torch = time.time() - start_time_torch

# Benchmark JAX
start_time_jax = time.time()
result_jax = compute_x_tilde_jax(
    inputs_jax, (eig_vals_jax, eig_vecs_jax)
).block_until_ready()
time_jax = time.time() - start_time_jax

# Move PyTorch result back to CPU for comparison
result_torch = result_torch.cpu()

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (JAX): {time_jax:.6f}s')

# Compare the results
print('\nComparing the results...')

if np.allclose(result_torch.numpy(), result_jax, atol=1e-4):
    print('The results from PyTorch and JAX are close enough.')
else:
    print(
        'The results from PyTorch and JAX differ more than the acceptable tolerance.'
    )

    # Find the indices where the results differ
    diff_indices = np.where(np.abs(result_torch.numpy() - result_jax) > 1e-4)

    # Print the differing indices and values
    print('Differing indices and values:')
    for i in range(len(diff_indices[0])):
        index = tuple(diff_index[i] for diff_index in diff_indices)
        print(f'Index: {index}')
        print(f'PyTorch value: {result_torch.numpy()[index]}')
        print(f'JAX value: {result_jax[index]}')
        print()
        if i >= 5:
            break