import torch
import jax
import jax.numpy as jnp
import numpy as np
import torch.autograd.profiler as profiler


# Test configurations
TEST_TORCH_COMPUTE_Y_T_TORCH = True
TEST_TORCH_COMPUTE_Y_T_STACK = True
TEST_TORCH_COMPUTE_Y_T_PSCAN_NAIVE = False
TEST_TORCH_COMPUTE_Y_T_PSCAN_FFT = False
TEST_JAX = True
PROFILE_TORCH = True
BENCHMARK_TORCH = True
COMPARE_RESULTS = True

# Test parameters
D_OUT = 128
K = 24
SEQ_LEN = 1000
ATOL = 1e-3


@torch.jit.script
def pscan_fft_simple(A, X):
    N, T, D = X.shape
    device = X.device
    A_log = torch.log(A.to(dtype=torch.cfloat))
    A_log_T = A_log.T
    A_log_T = torch.cat([A_log_T, torch.zeros(T - 1, N, device=device)], dim=0)
    mask1 = torch.where(
        (torch.arange(2 * T - 1, device=device) <= T - 1),
        1,
        0,
    )
    mask1 = mask1.unsqueeze(1)
    Z1_log_rev = torch.fft.ifft(
        torch.fft.fft(mask1, dim=0) * torch.fft.fft(A_log_T, dim=0),
        n=2 * T - 1,
        dim=0,
    )
    Z1_log_rev = Z1_log_rev[:T, :].T.unsqueeze(-1)
    mask2 = torch.where(
        torch.cat(
            [
                (
                    (torch.arange(2 * T - 1, device=device) >= 1)
                    & (torch.arange(2 * T - 1, device=device) <= t)
                ).unsqueeze(0)
                for t in range(T)
            ],
            dim=0,
        ),
        1,
        0,
    )
    mask2 = mask2.unsqueeze(-1)
    Z2_log_rev = torch.fft.ifft(
        torch.fft.fft(mask2, dim=1)
        * torch.fft.fft(A_log_T.unsqueeze(0), dim=1),
        n=2 * T - 1,
        dim=1,
    )
    Z2_log_rev = Z2_log_rev[:, :T, :]
    Z2_log_rev = Z2_log_rev.permute(2, 0, 1)
    Z2_log_rev = torch.tril(Z2_log_rev, diagonal=0)
    Z_log = Z1_log_rev - Z2_log_rev
    Z = torch.tril(torch.exp(Z_log), diagonal=0)
    Y_ = torch.bmm(Z, X)
    Y_ = torch.cat([torch.zeros(N, 1, D, device=device), Y_[:, :-1, :]], dim=1)
    Y = Y_ + X
    return Y


@torch.jit.script
def pscan_naive(A, X):
    N, T, D = X.shape
    device = X.device
    Y = torch.zeros(N, T, D, device=device, dtype=X.dtype)
    Y[:, 0, :] = X[:, 0, :]
    for k in range(1, X.shape[1]):
        Y[:, k, :] = A[:, k - 1].unsqueeze(1) * Y[:, k - 1, :] + X[:, k, :]
    return Y


@torch.jit.script
def get_eig(A):
    return torch.linalg.eig(A)


# @torch.jit.script
def compute_y_t_pscan(
    m_y: torch.Tensor, deltas: torch.Tensor, use_fft: bool
) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [seq_len, d_out].
    """

    # stack states and transitions to have a bona fide LDS. state is of dimension `k * d_out`
    # transition matrix has (a permuted) m_y in the first row, and identity on the -1 off-diagonal
    _, k, d = m_y.shape
    L, _ = deltas.shape
    device = m_y.device
    identity = torch.cat(
        [torch.eye((k - 1) * d), torch.zeros((k - 1) * d, d)], axis=1
    ).to(device)
    reshaped_M = m_y.reshape(d, k * d)
    A = torch.cat([reshaped_M, identity], axis=0)
    X = torch.cat([deltas, torch.zeros(L, (k - 1) * d, device=device)], axis=-1)

    # diagonalize dynamics and project
    evals, evecs = get_eig(A)
    inv_evecs = torch.linalg.inv(evecs)
    X_diag = X.to(torch.complex64) @ inv_evecs.T

    # run diagonal pscan. N = k * d and T = L
    A_pscan = torch.tile(evals[:, None], (1, L))
    X_pscan = X_diag.T[:, :, None]
    Y_pscan = (
        pscan_fft_simple(A_pscan, X_pscan)
        if use_fft
        else pscan_naive(A_pscan, X_pscan)
    )
    ys = Y_pscan[:, :, 0].T

    # undiagonalize
    ys = (ys @ evecs.T).to(torch.float32)
    return ys[:, :d]


# @torch.jit.script
def compute_y_t_stack(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [seq_len, d_out].
    """

    # stack states and transitions to have a bona fide LDS. state is of dimension `k * d_out`
    # transition matrix has (a permuted) m_y in the first row, and identity on the -1 off-diagonal
    # Extract the dimensions from the input tensors
    
    # _, k, d = m_y.shape
    # L, _ = deltas.shape

    # # Get the device of the input tensor m_y
    # device = m_y.device

    # # Create an identity matrix of size ((k - 1) * d, (k - 1) * d) on the specified device
    # identity_matrix = torch.eye((k - 1) * d, device=device)

    # # Create a zero matrix of size ((k - 1) * d, d) on the specified device
    # zero_matrix = torch.zeros((k - 1) * d, d, device=device)

    # # Concatenate the identity matrix and zero matrix along dimension 1
    # identity = torch.cat([identity_matrix, zero_matrix], dim=1)

    # # Reshape m_y to have dimensions [d, k * d]
    # reshaped_M = m_y.reshape(d, k * d)

    # # Concatenate the reshaped m_y and identity matrix along dimension 0
    # A = torch.cat([reshaped_M.to(device), identity], dim=0)

    # # Create a zero matrix of size [L, (k - 1) * d] on the specified device
    # zero_matrix_X = torch.zeros(L, (k - 1) * d, device=device)

    # # Concatenate the deltas and zero matrix along the last dimension
    # X = torch.cat([deltas, zero_matrix_X], dim=-1)

    # # Initialize y with the first row of X
    # y = X[0]

    # # Initialize a list to store the computed y values
    # ys = [y[:d]]

    # # Iterate over the remaining rows of X
    # for i in range(1, L):
    #     # Compute the next y value using matrix multiplication and addition
    #     y = A @ y + X[i]

    #     # Append the first d elements of y to the ys list
    #     ys.append(y[:d])

    # # Stack the computed y values along a new dimension
    # return torch.stack(ys)
    # Extract the dimensions from the input tensors
    _, k, d = m_y.shape
    L, _ = deltas.shape

    # Get the device of the input tensor m_y
    device = m_y.device

    # Concatenate m_y reshaped as [d, k*d] with an identity matrix to form matrix A
    A = torch.cat([
        m_y.reshape(d, k * d).to(device),
        torch.eye((k - 1) * d, k * d, device=device)  # Directly create the right size and shape
    ], dim=0)

    # Prepare input X with deltas followed by zeros for lower dimensions
    X = torch.cat([
        deltas,
        torch.zeros(L, (k - 1) * d, device=device)
    ], dim=1)

    # Initialize y and storage for results
    y = X[0]
    ys = [y[:d]]  # Store only the necessary part of y

    # Matrix-vector multiplication in a loop to update y
    for x in X[1:]:
        y = A @ y + x
        ys.append(y[:d])

    # Convert list to tensor
    return torch.stack(ys)


@torch.jit.script
def compute_y_t_torch(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Compute sequence of y_t given a series of deltas and m_y via a simple scan.

    Args:
        m_y (torch.Tensor): A matrix of shape [d_out, k, d_out] that acts as windowed
            transition matrix for the linear dynamical system evolving y_t.
        deltas (torch.Tensor): A matrix of shape [seq_len, d_out].

    Returns:
        torch.Tensor: A matrix of shape [seq_len, d_out].
    """
    d_out, k, _ = m_y.shape
    carry = torch.zeros((k, d_out), device=deltas.device)
    ys = torch.zeros((len(deltas), d_out), device=deltas.device)

    for i, x in enumerate(deltas):
        output = torch.tensordot(m_y, carry, dims=2) + x
        ys[i] = output
        carry = torch.roll(carry, shifts=1, dims=0)
        carry[0] = output

    return ys


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


def main():
    import time

    # Set a seed
    np.random.seed(1337)

    # Create random input data
    m_y_np = np.random.rand(D_OUT, K, D_OUT).astype(np.float32)
    deltas_np = np.random.rand(SEQ_LEN, D_OUT).astype(np.float32)
    m_y_torch = torch.from_numpy(m_y_np)
    deltas_torch = torch.from_numpy(deltas_np)
    m_y_jax = jnp.array(m_y_np)
    deltas_jax = jnp.array(deltas_np)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m_y_torch = m_y_torch.to(device)
    deltas_torch = deltas_torch.to(device)

    # Warm up the JIT compilers
    if TEST_TORCH_COMPUTE_Y_T_TORCH:
        print('Warming up the JIT compiler for COMPUTE_Y_T_TORCH...')
        _ = compute_y_t_torch(m_y_torch, deltas_torch)
        torch.cuda.synchronize()
    if TEST_TORCH_COMPUTE_Y_T_STACK:
        print('Warming up the JIT compiler for COMPUTE_Y_T_STACK...')
        _ = compute_y_t_stack(m_y_torch, deltas_torch)
        torch.cuda.synchronize()
    if TEST_TORCH_COMPUTE_Y_T_PSCAN_NAIVE:
        print('Warming up the JIT compiler for COMPUTE_Y_T_PSCAN (NO FFT)...')
        _ = compute_y_t_pscan(m_y_torch, deltas_torch, use_fft=False)
        torch.cuda.synchronize()
    if TEST_TORCH_COMPUTE_Y_T_PSCAN_FFT:
        print('Warming up the JIT compiler for COMPUTE_Y_T_PSCAN (FFT)...')
        _ = compute_y_t_pscan(m_y_torch, deltas_torch, use_fft=True)
        torch.cuda.synchronize()
    if TEST_JAX:
        print('Warming up the JIT compiler for COMPUTE_Y_T_JAX...')
        _ = compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()

    # PyTorch tests
    torch_versions = []
    if TEST_TORCH_COMPUTE_Y_T_TORCH:
        torch_versions.append(('compute_y_t_torch', compute_y_t_torch))
    if TEST_TORCH_COMPUTE_Y_T_STACK:
        torch_versions.append(('compute_y_t_stack', compute_y_t_stack))
    if TEST_TORCH_COMPUTE_Y_T_PSCAN_NAIVE:
        torch_versions.append(
            (
                'compute_y_t_pscan (naive)',
                lambda m_y, deltas: compute_y_t_pscan(
                    m_y, deltas, use_fft=False
                ),
            )
        )
    if TEST_TORCH_COMPUTE_Y_T_PSCAN_FFT:
        torch_versions.append(
            (
                'compute_y_t_pscan (FFT)',
                lambda m_y, deltas: compute_y_t_pscan(
                    m_y, deltas, use_fft=True
                ),
            )
        )

    results_torch = []
    execution_times = {}

    for version_name, compute_y_t_version in torch_versions:
        if PROFILE_TORCH:
            print(f'\nProfiling {version_name}...')
            with profiler.profile(with_stack=True, profile_memory=True) as prof:
                result_torch = compute_y_t_version(m_y_torch, deltas_torch)
                torch.cuda.synchronize()
            print(
                prof.key_averages(group_by_stack_n=5).table(
                    sort_by='cpu_time_total', row_limit=10
                )
            )

        if BENCHMARK_TORCH:
            print(f'Benchmarking {version_name}...')

            start_time_torch = time.time()

            result_torch = compute_y_t_version(m_y_torch, deltas_torch)

            torch.cuda.synchronize()

            time_torch = time.time() - start_time_torch

            execution_times[version_name] = time_torch

            print(f'Execution Time ({version_name}): {time_torch:.6f}s')

        results_torch.append((version_name, result_torch.cpu()))

    # JAX test

    if TEST_JAX:
        print('\nBenchmarking JAX...')

        start_time_jax = time.time()

        result_jax = compute_y_t_jax(m_y_jax, deltas_jax).block_until_ready()

        time_jax = time.time() - start_time_jax

        execution_times['JAX'] = time_jax

        print(f'Execution Time (JAX): {time_jax:.6f}s')

    # Compare the results

    if COMPARE_RESULTS and TEST_JAX and any(torch_versions):
        print('\nComparing the results...')

        for version_name, result_torch in results_torch:
            print(f'\nComparing {version_name} with JAX...')

            if np.allclose(result_torch.numpy(), result_jax, atol=ATOL):
                print(
                    f'The results from {version_name} and JAX are close enough.'
                )

            else:
                print(
                    f'The results from {version_name} and JAX differ more than the acceptable tolerance.'
                )

                # Find the indices where the results differ

                diff_indices = np.where(
                    np.abs(result_torch.numpy() - result_jax) > ATOL
                )

                # Print the differing indices and values

                print('Differing indices and values:')

                for i in range(len(diff_indices[0])):
                    index = tuple(diff_index[i] for diff_index in diff_indices)

                    print(f'Index: {index}')

                    print(
                        f'{version_name} value: {result_torch.numpy()[index]}'
                    )

                    print(f'JAX value: {result_jax[index]}')

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
