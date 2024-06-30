import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def get_hankel(n: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Generates a Hankel matrix Z, as defined in the paper.

    Note: This does not generate the Hankel matrix with the built-in
    negative featurization as mentioned in the appendix.

    Args:
        n (int): Size of the square Hankel matrix.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    i = torch.arange(1, n + 1)         # -> [n]
    s = i[:, None] + i[None, :]        # -> [n, n]
    Z = 2.0 / (s**3 - s + eps)         # -> [n, n]
    return Z


@torch.jit.script
def get_hankel_L(n: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Generates an alternative Hankel matrix Z_L that offers built-in
    negative featurization as mentioned in the appendix.

    Args:
        n (int): Size of the square Hankel matrix.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Hankel matrix Z of shape [n, n].
    """
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")

    # Calculate (-1)^(i+j-2) + 1
    sgn = (-1) ** (i + j - 2) + 1

    # Calculate the denominator
    denom = (i + j + 3) * (i + j - 1) * (i + j + 1)

    # Combine all terms
    Z_L = sgn * (8 / (denom + eps))

    return Z_L
    
# Optimized function
def get_hankel_L_optimized(n: int, eps: float = 1e-8) -> torch.Tensor:
    i = torch.arange(n, device=device).unsqueeze(1)  # Column vector
    j = torch.arange(n, device=device).unsqueeze(0)  # Row vector
    s = i + j
    sgn = (-1) ** (s - 2) + 1
    denom = (s + 3) * (s - 1) * (s + 1)
    Z_L = sgn * (8 / (denom + eps))
    return Z_L

# Run and time each function, and check for issues
def test_function(func, n):
    torch.cuda.synchronize()  # Wait for CUDA to finish all prior tasks
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    result = func(n)
    end.record()
    
    torch.cuda.synchronize()  # Wait for all kernels to finish
    time_ms = start.elapsed_time(end)  # Measure time in milliseconds
    
    has_inf_or_nan = torch.isinf(result).any() or torch.isnan(result).any()
    
    return time_ms, has_inf_or_nan, result

n = 500  # Size of the Hankel matrix

# Test each function
time_original, issues_original, _ = test_function(get_hankel_L, n)
time_optimized, issues_optimized, _ = test_function(get_hankel_L_optimized, n)
time_jit, issues_jit, _ = test_function(get_hankel, n)

print("Original function time: {:.2f}ms, Issues: {}".format(time_original, issues_original))
print("Optimized function time: {:.2f}ms, Issues: {}".format(time_optimized, issues_optimized))
print("JIT function time: {:.2f}ms, Issues: {}".format(time_jit, issues_jit))
