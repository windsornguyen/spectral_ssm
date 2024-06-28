# import jax
# import jax.numpy as jnp
# import jax.scipy.signal
# from jax import random
# from jax import vmap

# # Define the non-batched convolution function
# def conv_non_batched(v, u):
#     tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method='fft')[:x.shape[0]]
#     mvconv = vmap(tr_conv, in_axes=(1, None), out_axes=1)
#     mmconv = vmap(mvconv, in_axes=(None, 1), out_axes=-1)
#     return mmconv(v, u)

# # Define the batched convolution function
# def conv_batched(v, u):
#     tr_conv = lambda x, y: jax.scipy.signal.convolve(x, y, method='fft')[:x.shape[0]]
#     mvconv = vmap(tr_conv, in_axes=(1, None), out_axes=1)
#     mmconv = vmap(mvconv, in_axes=(None, 1), out_axes=-1)
#     bmmconv = vmap(mmconv, in_axes=(None, 0), out_axes=0)
#     out = bmmconv(v, u)
#     print(out.shape)
#     return out

# # Test data
# l, k, d_in, bsz = 29, 3, 14, 5

# key = random.PRNGKey(0)  # Initialize the random key
# v = random.normal(key, shape=(l, k))  # Generate random values for v
# key, subkey = random.split(key)  # Split the key to reuse
# u_single = random.normal(subkey, shape=(l, d_in))  # Generate random values for u_single
# u_batched = jnp.tile(u_single, (bsz, 1, 1))  # Create a batch where each slice is identical

# # Run the functions
# output_non_batched = conv_non_batched(v, u_single)
# output_batched = conv_batched(v, u_batched)

# # Check if outputs are identical
# identical = jnp.allclose(output_non_batched, output_batched[0], atol=1e-5)  # Compare the first batch output to the non-batched output
# print("Outputs are identical across all batches:", jnp.all(jnp.allclose(output_batched, output_batched[0], atol=1e-5)))  # Check across all batches
# print("Output comparison to non-batched version:", identical)

import torch
import jax
import numpy as np
import jax.numpy as jnp
from jax.scipy.signal import convolve as jax_convolve
import time
import torch.nn.functional as F

# Ensure PyTorch uses CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tr_conv_fft(x, y):
    """
    Perform convolution using FFT in PyTorch.

    Args:
        x (torch.Tensor): Input tensor of shape (seq_len,).
        y (torch.Tensor): Input tensor of shape (seq_len,).

    Returns:
        torch.Tensor: Convolution result of shape (seq_len,).
    """
    n = x.shape[0] + y.shape[0] - 1
    X = torch.fft.rfft(x, n=n)
    Y = torch.fft.rfft(y, n=n)
    Z = X * Y
    z = torch.fft.irfft(Z, n=n)
    return z[: x.shape[0]]


def tr_conv_direct(x, y):
    """
    Perform convolution using direct method in PyTorch.

    Args:
        x (torch.Tensor): Input tensor of shape (seq_len,).
        y (torch.Tensor): Input tensor of shape (seq_len,).

    Returns:
        torch.Tensor: Convolution result of shape (seq_len,).
    """
    n = x.shape[0] + y.shape[0] - 1

    # Perform FFT on both inputs with the calculated length
    x_fft = torch.fft.rfft(x, n=n)
    y_fft = torch.fft.rfft(y, n=n)

    # Element-wise multiplication in the frequency domain
    output_fft = x_fft * y_fft

    # Inverse FFT to return to the time domain
    output = torch.fft.irfft(output_fft, n=n)

    # Truncate to the original sequence length
    return output[: x.shape[0]]


def tr_conv_old(v, u):
    # Set the device
    device = v.device

    # Calculate the sequence length and determine the target tensor
    seq_len = max(v.size(0), u.size(0))
    target_tensor = torch.tensor(
        2 * seq_len - 1, device=device, dtype=torch.float32
    )

    # Calculate the ceiling of the log base 2 of the target tensor
    ceil_log_base_2 = torch.ceil(torch.log2(target_tensor))

    # Calculate the padded length as the next power of two
    padded_len = int(2**ceil_log_base_2)

    # Padding for FFT efficiency (lengths that are powers of two perform best)
    v_padded = F.pad(v, (0, padded_len - seq_len))
    u_padded = F.pad(u, (0, padded_len - seq_len))

    # Perform FFT on both padded inputs
    v_fft = torch.fft.rfft(v_padded)
    u_fft = torch.fft.rfft(u_padded)

    # Element-wise multiplication in the frequency domain
    output_fft = v_fft * u_fft

    # Inverse FFT to return to the time domain
    output = torch.fft.irfft(output_fft, n=padded_len)

    # Truncate to the original sequence length
    return output[:seq_len]


def jax_tr_conv(x, y):
    """
    Perform convolution using FFT in JAX.

    Args:
        x (numpy.ndarray): Input array of shape (seq_len,).
        y (numpy.ndarray): Input array of shape (seq_len,).

    Returns:
        numpy.ndarray: Convolution result of shape (seq_len,).
    """
    return jax_convolve(x, y, method='fft')[: x.shape[0]]


def run_benchmarks(seq_len=8012, num_runs=100):
    """
    Run benchmarks for different convolution methods.

    Args:
        seq_len (int): Sequence length for input tensors. Default is 4096.
        num_runs (int): Number of times to run each method. Default is 10.
    """
    x = torch.randn(seq_len, device=device)
    y = torch.randn(seq_len, device=device)

    methods = {
        'PyTorch FFT Convolution': tr_conv_fft,
        'PyTorch Direct Convolution': tr_conv_direct,
        'PyTorch Old': tr_conv_old,
    }

    # Warm-up runs for fair GPU timing
    for name, func in methods.items():
        _ = func(x, y)
    torch.cuda.synchronize()

    # Timings
    times = {name: [] for name in methods}
    for _ in range(num_runs):
        for name, func in methods.items():
            start_time = time.time()
            torch.cuda.synchronize()  # Start timing
            result = func(x, y)
            torch.cuda.synchronize()  # End timing
            times[name].append(time.time() - start_time)

    # Run JAX method
    x_np = np.array(x.cpu())
    y_np = np.array(y.cpu())
    times['JAX FFT Convolution'] = []
    for _ in range(num_runs):
        start_time = time.time()
        result_jax = jax_tr_conv(x_np, y_np)
        times['JAX FFT Convolution'].append(time.time() - start_time)

    # Check correctness
    result_ref = result_jax
    for name, func in methods.items():
        result_py = func(x, y).cpu().numpy()
        if not np.allclose(result_ref, result_py, atol=1e-4):
            print(f'{name} failed accuracy check.')
            print(f'Reference result: {result_ref}')
            print(f'{name} result: {result_py}')
            print(f'Difference: {np.abs(result_ref - result_py)}')
            print()

    # Calculate mean and standard deviation of timings
    mean_times = {name: np.mean(timings) for name, timings in times.items()}
    std_times = {name: np.std(timings) for name, timings in times.items()}

    # Print results sorted by mean time
    ranked_methods = sorted(mean_times.items(), key=lambda x: x[1])
    for rank, (name, mean_time) in enumerate(ranked_methods, start=1):
        std_time = std_times[name]
        print(f'{rank}. {name}: {mean_time:.6f} ± {std_time:.6f} seconds')


if __name__ == '__main__':
    run_benchmarks()
