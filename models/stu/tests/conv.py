import numpy as np
import torch
import jax
import jax.numpy as jnp
import jax.scipy.signal
import time
import math
from utils.nearest_power_of_2 import nearest_power_of_2


def tr_conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Perform truncated convolution using FFT.

    Args:
        v (torch.Tensor): Tensor of shape [seq_len,].
        u (torch.Tensor): Tensor of shape [seq_len,].

    Returns:
        torch.Tensor: Convolution result of shape [seq_len,].
    """
    n = x.shape[0] + y.shape[0] - 1
    X = torch.fft.rfft(x, n=n)
    Y = torch.fft.rfft(y, n=n)
    Z = X * Y
    z = torch.fft.irfft(Z, n=n)
    return z[: x.shape[0]]


def conv_torch_base(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis.
    This is the base reference implementation using explicit loops.

    Args:
        v (torch.Tensor): Top k eigenvectors of shape [sl, k].
        u (torch.Tensor): Input of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: A matrix of shape [bsz, sl, k, d_in].
    """
    bsz, l, d_in = u.shape
    k = v.shape[1]

    # Reshape and expand dimensions for broadcasting
    v = v.view(1, l, k, 1).expand(bsz, -1, -1, d_in)
    u = u.view(bsz, l, 1, d_in).expand(-1, -1, k, -1)

    # Perform convolution using tr_conv
    result = torch.zeros(bsz, l, k, d_in, device=v.device)
    for b in range(bsz):
        for i in range(k):
            for j in range(d_in):
                result[b, :, i, j] = tr_conv(v[b, :, i, j], u[b, :, i, j])

    return result


def conv_torch(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution to project input sequences into the spectral basis.

    Args:
        v (torch.Tensor): Top k eigenvectors of shape [sl, k].
        u (torch.Tensor): Input of shape [bsz, sl, d_in].

    Returns:
        torch.Tensor: A matrix of shape [bsz, sl, k, d_in].
    """
    bsz, sl, d_in = u.shape
    _, k = v.shape
    n = nearest_power_of_2(sl * 2 - 1)  # Round n to the nearest power of 2
    
    # Add and expand the bsz and d_in dims in v
    v = v.view(1, sl, k, 1)  # -> [1, sl, k, 1]
    v = v.expand(bsz, sl, k, d_in)  # -> [bsz, sl, k, d_in]

    # Add and expand the k dim in u
    u = u.view(bsz, sl, 1, d_in)  # -> [bsz, sl, 1, d_in]
    u = u.expand(bsz, sl, k, d_in)  # -> [bsz, sl, k, d_in]

    V = torch.fft.rfft(v, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    Z = V * U
    z = torch.fft.irfft(Z, n=n, dim=1)

    return z[:, :sl]


def spectral_conv(
    φ: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spectral convolution to project input sequences into the spectral basis.

    This implements the computation of U⁺_t,k and U⁻_t,k as described in Section 3
    of the paper.

    Args:
        φ (torch.Tensor): Top K eigenvectors of shape [sl, K].
        u (torch.Tensor): Input of shape [bsz, sl, d_in].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: U⁺ and U⁻ of shape [bsz, sl, K, d_in].
    """
    bsz, sl, d_in = u.shape
    K = φ.shape[1]

    n = 2 ** math.ceil(math.log2(sl * 2 - 1))

    # Expand dimensions for broadcasting
    φ = φ.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, d_in)
    u = u.unsqueeze(2).expand(-1, -1, K, -1)

    # Compute U⁺
    Φ = torch.fft.rfft(φ, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    U_plus = torch.fft.irfft(Φ * U, n=n, dim=1)[:, :sl]

    # Compute U⁻
    u_alt = u * torch.tensor([1, -1], device=u.device).repeat((sl + 1) // 2)[
        :sl
    ].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    U_alt = torch.fft.rfft(u_alt, n=n, dim=1)
    U_minus = torch.fft.irfft(Φ * U_alt, n=n, dim=1)[:, :sl]

    return U_plus, U_minus


# JAX implementation using jax.jit and batch sizes
@jax.jit
def conv_jax(
    v: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    """Compute convolution to project input sequences into the spectral basis.

    Args:
        v: Top k eigenvectors of shape [l, k].
        u: Input of shape [bsz, l, d_in].

    Returns:
        A matrix of shape [bsz, l, k, d_in]
    """
    # Convolve two vectors of length l (x.shape[0]) and truncate to the l oldest
    # values.
    tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method="fft")[: x.shape[0]]

    # Convolve each sequence of length l in v with each sequence in u.
    mvconv = jax.vmap(tr_conv, in_axes=(1, None), out_axes=1)
    mmconv = jax.vmap(mvconv, in_axes=(None, 1), out_axes=-1)

    # Add an additional vmap to handle the batch dimension
    bmmconv = jax.vmap(mmconv, in_axes=(None, 0), out_axes=0)

    return bmmconv(v, u)


def main():
    # Set seed for reproducibility
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    batch_size, seq_len, k, d_in = (
        torch.randint(1, 8, (1,)).item(),
        torch.randint(1, 32, (1,)).item(),
        torch.randint(1, 8, (1,)).item(),
        torch.randint(1, 32, (1,)).item(),
    )

    v = np.random.rand(seq_len, k)
    u = np.random.rand(batch_size, seq_len, d_in)

    v_torch = torch.tensor(v, device=device, dtype=torch.float32)
    u_torch = torch.tensor(u, device=device, dtype=torch.float32)
    v_jax = jnp.array(v)
    u_jax = jnp.array(u)

    # Warm-up JIT compilation
    _ = spectral_conv(v_torch, u_torch)
    _ = conv_jax(v_jax, u_jax).block_until_ready()

    # Test and benchmark all implementations
    implementations = [
        ("PyTorch conv base", lambda: conv_torch_base(v_torch, u_torch)),
        # ("PyTorch tr_conv", lambda: tr_conv(v_torch, u_torch)),
        ("PyTorch conv", lambda: conv_torch(v_torch, u_torch)),
        ("PyTorch spectral_conv", lambda: spectral_conv(v_torch, u_torch)[0]),
        ("JAX conv", lambda: conv_jax(v_jax, u_jax)),
    ]

    results = {}
    times = {}

    for name, func in implementations:
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            output = func()
            end_event.record()

            torch.cuda.synchronize()
            execution_time = (
                start_event.elapsed_time(end_event) / 1000
            )  # Convert to seconds
        else:
            start_time = time.perf_counter()
            output = func()
            execution_time = time.perf_counter() - start_time

        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        elif isinstance(output, jnp.ndarray):
            output = np.array(output)

        results[name] = output
        times[name] = execution_time

    # Compare outputs
    reference = results["PyTorch conv"]
    for name, result in results.items():
        if name == "PyTorch conv":
            continue

        if result.shape != reference.shape:
            print(f"Shape mismatch for {name}: {result.shape} vs {reference.shape}")
            continue

        if not np.allclose(result, reference, atol=1e-5):
            print(f"Values differ for {name}")
            difference_matrix = np.abs(result - reference)
            it = np.nditer(difference_matrix, flags=["multi_index"])
            print("Differing Values:")
            i = 0
            while not it.finished:
                if it[0] > 1e-5:
                    idx = it.multi_index
                    print(
                        f"Index: {idx}, Reference: {reference[idx]}, {name}: {result[idx]}, Diff: {it[0]}"
                    )
                    i += 1
                    if i >= 10:
                        print("...")
                        break
                it.iternext()
        else:
            print(f"Outputs for {name} are sufficiently close to the reference.")

    # Output performance metrics
    print("\nExecution Times:")
    for name, execution_time in times.items():
        print(f"{name}: {execution_time:.6f}s")


if __name__ == "__main__":
    main()
