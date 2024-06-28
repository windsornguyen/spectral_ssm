import jax
import jax.numpy as jnp
import numpy as np
import torch
import functools
import time
from torch.profiler import profile as profiler

@jax.jit
def compute_ar_x_preds_jax(
    w: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the auto-regressive component of spectral SSM.
    Args:
        w: An array of shape [d_out, d_in, k].
        x: A single input sequence of shape [l, d_in].
    Returns:
        ar_x_preds: An output of shape [l, d_out].
    """
    d_out, _, k = w.shape
    l = x.shape[0]
    # Contract over `d_in`.
    o = jnp.einsum('oik,li->klo', w, x)
    # For each `i` in `k`, roll the `(l, d_out)` by `i` steps.
    o = jax.vmap(functools.partial(jnp.roll, axis=0))(o, jnp.arange(k))
    # Create a mask that zeros out nothing at `k=0`, the first `(l, d_out)` at
    # `k=1`, the first two `(l, dout)`s at `k=2`, etc.
    m = jnp.triu(jnp.ones((k, l)))
    m = jnp.expand_dims(m, axis=-1)
    m = jnp.tile(m, (1, 1, d_out))
    # Mask and sum along `k`.
    return jnp.sum(o * m, axis=0)

@jax.jit
def compute_ar_x_preds_jax_bsz(
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

# Prepare random data for testing
bsz = np.random.randint(1, 10)
d_out = np.random.randint(10, 100)
k = np.random.randint(1, 10)
seq_len = np.random.randint(100, 1000)
print(f'Testing compute_ar_x_preds function for w of shape [{d_out}, {d_out}, {k}] and x of shape [{seq_len}, {d_out}].')
print(f'Testing compute_ar_x_preds_bsz function for w of shape [{bsz}, {d_out}, {d_out}, {k}] and x of shape [{bsz}, {seq_len}, {d_out}].')

# Create random input data
w_np = np.random.rand(d_out, d_out, k).astype(np.float32)
x_np = np.random.rand(seq_len, d_out).astype(np.float32)
w_jax = jnp.array(w_np)
x_jax = jnp.array(x_np)

x_bsz_np = np.tile(x_np[np.newaxis, :, :], (bsz, 1, 1))
w_bsz_jax = jnp.array(w_np)
x_bsz_jax = jnp.array(x_bsz_np)

# Warm up the JIT compilers
_ = compute_ar_x_preds_jax(w_jax, x_jax)
_ = compute_ar_x_preds_jax_bsz(w_bsz_jax, x_bsz_jax)

# Benchmark JAX
start_time_jax = time.time()
result_jax = compute_ar_x_preds_jax(w_jax, x_jax)
time_jax = time.time() - start_time_jax

start_time_jax_bsz = time.time()
result_jax_bsz = compute_ar_x_preds_jax_bsz(w_bsz_jax, x_bsz_jax)
time_jax_bsz = time.time() - start_time_jax_bsz

# Output performance metrics
print(f'\nExecution Time (JAX): {time_jax:.6f}s')
print(f'Execution Time (JAX with bsz): {time_jax_bsz:.6f}s')

# Compare the results
print('\nComparing the results...')
for i in range(bsz):
    if np.allclose(result_jax, result_jax_bsz[i], atol=1e-8):
        print(f'The results from JAX and JAX with bsz (batch {i}) are close enough.')
    else:
        print(f'The results from JAX and JAX with bsz (batch {i}) differ more than the acceptable tolerance.')
        
        # Find the indices where the results differ
        diff_indices = np.where(np.abs(result_jax - result_jax_bsz[i]) > 1e-8)
        
        # Print the differing indices and values
        print('Differing indices and values:')
        for j in range(len(diff_indices[0])):
            index = tuple(diff_index[j] for diff_index in diff_indices)
            print(f'Index: {index}')
            print(f'JAX value: {result_jax[index]}')
            print(f'JAX with bsz value: {result_jax_bsz[i][index]}')
            print()
            if j >= 5:
                break