import torch
import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import numpy as np
import time

GLOBAL_SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "VECSIZE": 2}),
        triton.Config({"BLOCK_SIZE": 256, "VECSIZE": 2}),
        triton.Config({"BLOCK_SIZE": 512, "VECSIZE": 2}),
        triton.Config({"BLOCK_SIZE": 1024, "VECSIZE": 2}),
    ],
    key=["N_ROWS"],
)
@triton.jit
def roll_zero_kernel(
    src_ptr, dst_ptr, 
    N_ROWS: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    VECSIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Vectorized load
    row = offsets // VECSIZE
    col = offsets % VECSIZE
    rolled_row = (row + 1) % N_ROWS
    rolled_offsets = rolled_row * VECSIZE + col
    
    mask = offsets < N_ROWS * VECSIZE
    x = tl.load(src_ptr + rolled_offsets, mask=mask, other=0.0)
    
    # Zero first row
    x = tl.where(row == 0, 0.0, x)
    
    # Vectorized store
    tl.store(dst_ptr + offsets, x, mask=mask)

def triton_roll_and_zero(x):
    assert x.size(1) == 2, "Input must have 2 columns"
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    roll_zero_kernel[grid](x, y, N_ROWS=x.size(0))
    return y

def torch_roll_and_zero(x, k=1):
    shifted = torch.roll(x, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted

@triton.jit
def combine_fn(a, b):
    return a + b

def compute_y_t_triton(m_y, deltas):
    d_out, k, _ = m_y.shape

    @triton.jit
    def scan_op(carry, x):
        output = tl.dot(m_y, carry, axes=2) + x
        carry = triton_roll_and_zero(carry)
        return carry, output

    initial_carry = torch.zeros((k, d_out), device="cuda")
    _, ys = tl.associative_scan(scan_op, initial_carry, deltas)
    return ys

# JAX implementation
@jax.jit
def compute_y_t_jax(m_y: jnp.ndarray, deltas: jnp.ndarray) -> jnp.ndarray:
    d_out, k, _ = m_y.shape

    def scan_op(carry, x):
        output = jnp.tensordot(m_y, carry, axes=2) + x
        carry = jnp.roll(carry, 1, axis=0)
        carry = carry.at[0].set(output)
        return carry, output

    _, ys = jax.lax.scan(scan_op, jnp.zeros((k, d_out)), deltas)
    return ys

# Comparison function
def compare_implementations(d_out, k, seq_len, num_runs=100):
    # Generate random inputs
    torch.manual_seed(1337)
    m_y_torch = torch.randn(d_out, k, d_out, device="cuda")
    deltas_torch = torch.randn(seq_len, d_out, device="cuda")

    m_y_jax = jnp.array(m_y_torch.cpu().numpy())
    deltas_jax = jnp.array(deltas_torch.cpu().numpy())

    # Warm-up runs
    for _ in range(10):
        compute_y_t_triton(m_y_torch, deltas_torch)
        compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()

    # PyTorch timing
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        ys_torch = compute_y_t_triton(m_y_torch, deltas_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / num_runs

    # JAX timing
    start_time = time.time()
    for _ in range(num_runs):
        ys_jax = compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()
    jax_time = (time.time() - start_time) / num_runs

    # Check results
    ys_torch_cpu = ys_torch.cpu().numpy()
    ys_jax_cpu = np.array(ys_jax)
    is_close = np.allclose(ys_torch_cpu, ys_jax_cpu, rtol=1e-5, atol=1e-5)

    print(f"Inputs: d_out={d_out}, k={k}, seq_len={seq_len}")
    print(f"Results are close: {is_close}")
    print(f"PyTorch time: {torch_time:.6f} seconds")
    print(f"JAX time: {jax_time:.6f} seconds")
    print(f"Speed ratio (JAX/PyTorch): {jax_time/torch_time:.2f}")

# Run comparisons
torch.set_float32_matmul_precision("high")
set_seed(GLOBAL_SEED)
print("Comparison 1:")
compare_implementations(d_out=2, k=2, seq_len=2)
