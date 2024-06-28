import torch
import jax
import jax.numpy as jnp
import numpy as np
import time
import functools
import torch.autograd.profiler as profiler

def compute_ar_x_preds_torch(
    w: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the auto-regressive component of spectral SSM.

    Args:
        w (torch.Tensor): A weight matrix of shape [d_out, d_in, k].
        x (torch.Tensor): Batch of input sequences of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: ar_x_preds: An output of shape [bsz, sl, d_out].
    """
    # bsz, l, d_in = x.shape
    # d_out, _, k = w.shape

    # # Contract over `d_in`
    # o = torch.einsum('oik,bli->bklo', w, x)  # [bsz, k, l, d_out]

    # # For each `i` in `k`, roll the `(l, d_out)` by `i` steps for each batch
    # rolled_o = torch.stack([torch.roll(o[:, i], shifts=i, dims=1) for i in range(k)], dim=1)

    # # Create a mask that zeros out nothing at `k=0`, the first `(l, d_out)` at
    # # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
    # m = torch.triu(torch.ones((k, l), device=w.device))
    # m = m.unsqueeze(-1).unsqueeze(0).expand(bsz, k, l, d_out)

    # # Mask and sum along `k`
    # return torch.sum(rolled_o * m, dim=1)
    bsz, sl, d_in = x.shape
    d_out, _, k = w.shape

    # Contract over `d_in` to combine weights with input sequences
    o = torch.einsum('oik,bli->bklo', w, x)  # [bsz, k, l, d_out]

    # For each `i` in `k`, shift outputs by `i` positions to align for summation.
    rolled_o = torch.stack(
        [torch.roll(o[:, i], 
        shifts=i, 
        dims=1
    ) for i in range(k)], dim=1) # -> [bsz, k, l, d_out]

    # Create a mask that zeros out nothing at `k=0`, the first `(sl, d_out)` at
    # `k=1`, the first two `(sl, dout)`s at `k=2`, etc.
    mask = torch.triu(torch.ones((k, sl), device=w.device))
    mask = mask.view(k, sl, 1) # Add d_out dim: -> [k, sl, 1]

    # Apply the mask and sum along `k`
    return torch.sum(rolled_o * mask, dim=1)


@jax.jit
def compute_ar_x_preds_jax(
    w: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the auto-regressive component of spectral SSM with batch size.
    Args:
        w: An array of shape [d_out, d_in, k].
        x: A batch of input sequences of shape [bsz, l, d_in].
    Returns:
        ar_x_preds: An output of shape [bsz, l, d_out].
    """
    bsz, l, d_in = x.shape
    d_out, _, k = w.shape

    # Contract over `d_in`.
    o = jnp.einsum('oik,bli->bklo', w, x)

    # For each `i` in `k`, roll the `(l, d_out)` by `i` steps for each batch.
    o = jax.vmap(lambda o_slice: jax.vmap(functools.partial(jnp.roll, axis=0))(o_slice, jnp.arange(k)))(o)
    
    # Create a mask that zeros out nothing at `k=0`, the first `(l, d_out)` at
    # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
    m = jnp.triu(jnp.ones((k, l)))
    m = jnp.expand_dims(m, axis=-1)
    m = jnp.expand_dims(m, axis=0)  # Expand for the batch dimension
    m = jnp.tile(m, (bsz, 1, 1, d_out))
    
    # Mask and sum along `k`.
    return jnp.sum(o * m, axis=1)


# Set a seed
np.random.seed(1337)
torch.manual_seed(1337)

# Prepare random data for testing
bsz = np.random.randint(1, 16)
d_out = np.random.randint(1, 37)
k = np.random.randint(1, 24)
seq_len = np.random.randint(1, 1000)
print(f'Testing compute_ar_x_preds function for w of shape [{d_out}, {d_out}, {k}] and x of shape [{seq_len}, {d_out}].')

# Create random input data
w_np = np.random.rand(d_out, d_out, k).astype(np.float32)
x_np = np.random.rand(bsz, seq_len, d_out).astype(np.float32)
w_torch = torch.from_numpy(w_np)
x_torch = torch.from_numpy(x_np)
w_jax = jnp.array(w_np)
x_jax = jnp.array(x_np)

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
w_torch = w_torch.to(device)
x_torch = x_torch.to(device)

# Warm up the JIT compilers
_ = compute_ar_x_preds_torch(w_torch, x_torch)
_ = compute_ar_x_preds_jax(w_jax, x_jax)

# Profile PyTorch
with profiler.profile(with_stack=True, profile_memory=True) as prof:
    _ = compute_ar_x_preds_torch(w_torch, x_torch)

# Print the profiling results for PyTorch
print(prof.key_averages().table(sort_by="cpu_time_total"))

# Benchmark PyTorch
start_time_torch = time.time()
result_torch = compute_ar_x_preds_torch(w_torch, x_torch)
time_torch = time.time() - start_time_torch

# Benchmark JAX
start_time_jax = time.time()
result_jax = compute_ar_x_preds_jax(w_jax, x_jax)
time_jax = time.time() - start_time_jax

# Move PyTorch result back to CPU for comparison
result_torch = result_torch.cpu()

# Output performance metrics
print(f'\nExecution Time (PyTorch): {time_torch:.6f}s')
print(f'Execution Time (JAX): {time_jax:.6f}s')

# Compare the results
print('\nComparing the results...')

if np.allclose(result_torch.numpy(), result_jax, atol=1e-8):
    print('The results from PyTorch and JAX are close enough.')
else:
    print('The results from PyTorch and JAX differ more than the acceptable tolerance.')
    
    # Find the indices where the results differ
    diff_indices = np.where(np.abs(result_torch.numpy() - result_jax) > 1e-8)
    
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
