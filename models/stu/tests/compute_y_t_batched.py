import torch
import jax
import jax.numpy as jnp
import numpy as np
import torch.autograd.profiler as profiler


# Test configurations
TEST_TORCH_COMPUTE_Y_T_TORCH_BATCHED = True
TEST_TORCH_COMPUTE_Y_T_STACK_BATCHED = True
TEST_JAX_BATCHED = True
TEST_JAX = False
PROFILE_TORCH = True
BENCHMARK_TORCH = True
COMPARE_RESULTS = True

# Test parameters
D_OUT = 29
K = 2
BATCH_SIZE = 4
SEQ_LEN = 10
ATOL = 1e-3


@torch.jit.script
def compute_y_t_torch_batched(
    m_y: torch.Tensor, deltas: torch.Tensor
) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan with batch size.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [bsz, seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [bsz, seq_len, d_out].
    """
    # d_out, k, _ = m_y.shape
    # bsz, seq_len, _ = deltas.shape
    # carry = torch.zeros((bsz, d_out, k), device=deltas.device)
    # ys = torch.zeros((bsz, seq_len, d_out), device=deltas.device)

    # for i in range(seq_len):
    #     output = torch.einsum('ijk,bkj->bi', m_y, carry) + deltas[:, i, :]
    #     ys[:, i, :] = output
    #     carry = torch.roll(carry, shifts=1, dims=2)
    #     carry[:, :, 0] = output

    # return ys
    # d_out, k, _ = m_y.shape
    # bsz, seq_len, _ = deltas.shape
    
    # # Create a tensor of shape (bsz, seq_len, d_out, k) with shifted deltas
    # shifted_deltas = torch.cat([
    #     torch.zeros((bsz, 1, d_out, k), device=deltas.device),
    #     deltas.unfold(1, k, 1)
    # ], dim=1)
    
    # # Compute the output using broadcasting and einsum
    # ys = torch.einsum('ijk,bljk->bli', m_y, shifted_deltas).sum(dim=-1)

    # return ys
    d_out, k, _ = m_y.shape
    bsz, sl, _ = deltas.shape

    # Define the transition matrix A, and add bsz for bmm
    A = m_y.view(d_out, k * d_out)  # Reshape m_y to [d_out, k * d_out] for concat
    eye = torch.eye((k - 1) * d_out, k * d_out, dtype=deltas.dtype, device=deltas.device)
    A = torch.cat([A, eye], dim=0)
    A = A.unsqueeze(0).expand(bsz, k * d_out, k * d_out) # -> [bsz, k * d_out, k * d_out]

    # Add (k - 1) rows of padding to deltas
    padding = torch.zeros(
        bsz, sl, (k - 1) * d_out, dtype=deltas.dtype, device=deltas.device
    ) # -> [bsz, sl, (k - 1) * d_out]

    carry = torch.cat([deltas, padding], dim=2)  # -> [bsz, sl, k * d_out]

    # Reshape for sequential processing
    carry = carry.view(bsz, sl, k * d_out, 1) # -> [bsz, sl, k * d_out, 1]

    # Initialize y and the output list of y's
    y = carry[:, 0]  # -> [bsz, k * d_out, 1]
    ys = [y[:, :d_out, 0]] # -> [bsz, d_out]

    # Iterate through the sequence
    # TODO: Unsure of how to further vectorize/optimize this given its sequential nature.
    # This loop takes up __98%__ of this function.
    for i in range(1, sl):
        y = torch.bmm(A, y) + carry[:, i]
        ys.append(y[:, :d_out, 0])
    ys = torch.stack(ys, dim=1) # -> [bsz, sl, d_out]
    
    return ys


@torch.jit.script
def compute_y_t_stack_batched(
    m_y: torch.Tensor, deltas: torch.Tensor
) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan with batch size.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [bsz, seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [bsz, seq_len, d_out].
    """
    d_out, k, _ = m_y.shape
    bsz, seq_len, _ = deltas.shape

    device = m_y.device

    A = torch.cat(
        [
            m_y.reshape(d_out, k * d_out).to(device),
            torch.eye(
                (k - 1) * d_out, k * d_out, device=device, dtype=torch.float32
            ),
        ],
        dim=0,
    )

    X = torch.cat(
        [
            deltas,
            torch.zeros(
                bsz,
                seq_len,
                (k - 1) * d_out,
                device=device,
                dtype=torch.float32,
            ),
        ],
        dim=2,
    )

    y = X[:, 0]
    ys = [y[..., :d_out]]

    for x in X[:, 1:].transpose(0, 1):
        y = A @ y.reshape(bsz, k * d_out, 1) + x.reshape(bsz, k * d_out, 1)
        ys.append(y[:, :d_out, 0])

    return torch.stack(ys, dim=1)


@jax.jit
def compute_y_t_jax(m_y: jnp.ndarray, deltas: jnp.ndarray) -> jnp.ndarray:
    """Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y: A matrix of shape [d_out, k, d_out] that acts as windowed transition
            matrix for the linear dynamical system evolving y_t.
        deltas: A matrix of shape [l, d_out].

    Returns:
        A matrix of shape [l, d_out].
    """
    d_out, k, _ = m_y.shape

    def scan_op(carry, x):
        output = jnp.tensordot(m_y, carry, axes=2) + x
        carry = jnp.roll(carry, 1, axis=0)
        carry = carry.at[0].set(output)
        return carry, output

    _, ys = jax.lax.scan(scan_op, jnp.zeros((k, d_out)), deltas)
    return ys


@jax.jit
def compute_y_t_jax_batched(
    m_y: jnp.ndarray, deltas: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan with batch size.

    Args:
        m_y: A matrix of shape [d_out, k, d_out] that acts as windowed transition
            matrix for the linear dynamical system evolving y_t.
        deltas: A matrix of shape [bsz, l, d_out].

    Returns:
        A matrix of shape [bsz, l, d_out].
    """
    return jax.vmap(compute_y_t_jax, in_axes=(None, 0), out_axes=0)(m_y, deltas)


def main():
    import time

    # Set a seed
    np.random.seed(1337)

    # Create random input data
    m_y_np = np.random.rand(D_OUT, K, D_OUT).astype(np.float32)
    deltas_np = np.random.rand(SEQ_LEN, D_OUT).astype(np.float32)
    deltas_bsz_np = np.random.rand(BATCH_SIZE, SEQ_LEN, D_OUT).astype(
        np.float32
    )
    m_y_torch = torch.from_numpy(m_y_np)
    deltas_torch = torch.from_numpy(deltas_np)
    deltas_bsz_torch = torch.from_numpy(deltas_bsz_np)
    m_y_jax = jnp.array(m_y_np)
    deltas_jax = jnp.array(deltas_np)
    deltas_bsz_jax = jnp.array(deltas_bsz_np)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m_y_torch = m_y_torch.to(device)
    deltas_torch = deltas_torch.to(device)
    deltas_bsz_torch = deltas_bsz_torch.to(device)

    print(
        'Testing compute_y_t_jax function for m_y of shape',
        m_y_np.shape,
        'and deltas of shape',
        deltas_np.shape,
    )
    print(
        'Testing compute_y_t_jax_batched function for m_y of shape',
        m_y_np.shape,
        'and deltas of shape',
        deltas_bsz_np.shape,
    )
    print('Shape of m_y_np:', m_y_np.shape)
    print('Shape of deltas_np:', deltas_np.shape)
    print('Shape of m_y_jax:', m_y_jax.shape)
    print('Shape of deltas_jax:', deltas_jax.shape)
    print('Shape of deltas_bsz_np:', deltas_bsz_np.shape)

    # Warm up the JIT compilers
    if TEST_TORCH_COMPUTE_Y_T_TORCH_BATCHED:
        print('Warming up the JIT compiler for COMPUTE_Y_T_TORCH_BATCHED...')
        _ = compute_y_t_torch_batched(m_y_torch, deltas_bsz_torch)
        torch.cuda.synchronize()
    if TEST_TORCH_COMPUTE_Y_T_STACK_BATCHED:
        print('Warming up the JIT compiler for COMPUTE_Y_T_STACK_BATCHED...')
        _ = compute_y_t_stack_batched(m_y_torch, deltas_bsz_torch)
        torch.cuda.synchronize()
    if TEST_JAX_BATCHED:
        print('Warming up the JIT compiler for COMPUTE_Y_T_JAX_BATCHED...')
        _ = compute_y_t_jax_batched(m_y_jax, deltas_bsz_jax).block_until_ready()
    if TEST_JAX:
        print('Warming up the JIT compiler for COMPUTE_Y_T_JAX...')
        _ = compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()

    # PyTorch tests
    torch_versions = []
    if TEST_TORCH_COMPUTE_Y_T_TORCH_BATCHED:
        torch_versions.append(
            ('compute_y_t_torch_batched', compute_y_t_torch_batched)
        )
    if TEST_TORCH_COMPUTE_Y_T_STACK_BATCHED:
        torch_versions.append(
            ('compute_y_t_stack_batched', compute_y_t_stack_batched)
        )

    results_torch = []
    execution_times = {}

    for version_name, compute_y_t_version in torch_versions:
        if PROFILE_TORCH:
            print(f'\nProfiling {version_name}...')
            with profiler.profile(with_stack=True, profile_memory=True) as prof:
                result_torch = compute_y_t_version(m_y_torch, deltas_bsz_torch)
                torch.cuda.synchronize()
            print(
                prof.key_averages(group_by_stack_n=5).table(
                    sort_by='cpu_time_total', row_limit=10
                )
            )

        if BENCHMARK_TORCH:
            print(f'Benchmarking {version_name}...')

            start_time_torch = time.time()

            result_torch = compute_y_t_version(m_y_torch, deltas_bsz_torch)

            torch.cuda.synchronize()

            time_torch = time.time() - start_time_torch

            execution_times[version_name] = time_torch

            print(f'Execution Time ({version_name}): {time_torch:.6f}s')

        results_torch.append((version_name, result_torch.cpu()))

    # JAX tests

    if TEST_JAX_BATCHED:
        print('\nBenchmarking JAX Batched...')

        start_time_jax_batched = time.time()

        result_jax_batched = compute_y_t_jax_batched(
            m_y_jax, deltas_bsz_jax
        ).block_until_ready()

        time_jax_batched = time.time() - start_time_jax_batched

        execution_times['JAX Batched'] = time_jax_batched

        print(f'Execution Time (JAX Batched): {time_jax_batched:.6f}s')

    if TEST_JAX:
        print('\nBenchmarking JAX...')

        start_time_jax = time.time()

        result_jax = compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()

        time_jax = time.time() - start_time_jax

        execution_times['JAX'] = time_jax

        print(f'Execution Time (JAX): {time_jax:.6f}s')

    # Compare the results

    if COMPARE_RESULTS and TEST_JAX_BATCHED and any(torch_versions):
        print('\nComparing the results...')

        for version_name, result_torch in results_torch:
            print(f'\nComparing {version_name} with JAX Batched...')

            if np.allclose(result_torch.numpy(), result_jax_batched, atol=ATOL):
                print(
                    f'The results from {version_name} and JAX Batched are close enough.'
                )

            else:
                print(
                    f'The results from {version_name} and JAX Batched differ more than the acceptable tolerance.'
                )

                # Find the indices where the results differ

                diff_indices = np.where(
                    np.abs(result_torch.numpy() - result_jax_batched) > ATOL
                )

                # Print the differing indices and values

                print('Differing indices and values:')

                for i in range(len(diff_indices[0])):
                    index = tuple(diff_index[i] for diff_index in diff_indices)

                    print(f'Index: {index}')

                    print(
                        f'{version_name} value: {result_torch.numpy()[index]}'
                    )

                    print(f'JAX Batched value: {result_jax_batched[index]}')

                    print()

                    if i >= 5:
                        break

    # Find the fastest version

    if execution_times:
        ranked_versions = sorted(execution_times.items(), key=lambda x: x[1])

        print('\nVersions ranked by execution time:')

        for i, (version, time) in enumerate(ranked_versions, start=1):
            print(f'{i}. {version}: {time:.6f}s')


if __name__ == '__main__':
    main()
