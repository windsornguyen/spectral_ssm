import torch
from prettytable import PrettyTable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1337)


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Shift the time axis by k steps to align u_{t-k} with u_t.

    This function implements the time shifting functionality needed for
    the autoregressive component in Equation 4 of the STU model (Section 3).

    Args:
        u (torch.Tensor): An input tensor of shape [bsz, sl, K, d],
        where bsz is the batch size, sl is the sequence length,
        K is the number of spectral filters, and d is the feature dimension.
        k (int): Number of time steps to shift. Defaults to 1.

    Returns:
        torch.Tensor: Shifted tensor of shape [bsz, sl, K, d].
    """
    if k == 0:  # Early return
        return u
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted


@torch.jit.script
def nearest_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than or equal to x. If x is already a power of 2,
    it returns x itself. Otherwise, it returns the next higher power of 2.

    Args:
        x (int): The input integer for which the nearest power of 2 is to be found.

    Returns:
        int: The smallest power of 2 that is greater than or equal to x.
    """
    s = bin(x)
    s = s.lstrip("-0b")
    length = len(s)
    return 1 << (length - 1) if x == 1 << (length - 1) else 1 << length



def conv(u: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FFT convolution of the input sequences into the Hankel spectral basis.

    This implements the computation of U⁺_{t,k} and U⁻_{t,k}, as described
    in Section 3 of the paper.

    Args:
        u (torch.Tensor): Input of shape [bsz, sl, d].
        phi (torch.Tensor): Top K eigenvectors of shape [sl, K].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Feature tensors of shape [bsz, sl, K, d].
    """
    bsz, sl, d = u.shape
    _, K = phi.shape

    # Round sequence length to the nearest power of 2 for efficient convolution
    n = nearest_power_of_2(sl * 2 - 1)

    # Add bsz and d dims to phi and u and expand to the return shape
    phi = phi.view(1, -1, K, 1).expand(bsz, -1, K, d)
    u = u.view(bsz, -1, 1, d).expand(bsz, -1, K, d)

    # Compute U⁺
    V = torch.fft.rfft(phi, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    U_plus = torch.fft.irfft(V * U, n=n, dim=1)[:, :sl]

    # Generate alternating signs tensor, (-1)^i of length sl, match dims of u
    alt = torch.ones(sl, device=u.device)
    alt[1::2] = -1  # Replace every other element with -1, starting from index 1
    alt = alt.view(1, sl, 1, 1).expand_as(u)

    # Compute U⁻
    u_alt = u * alt
    U_alt = torch.fft.rfft(u_alt, n=n, dim=1)
    U_minus = torch.fft.irfft(V * U_alt, n=n, dim=1)[:, :sl]

    return U_plus, U_minus


def compute_spectral(
    inputs: torch.Tensor,
    eigh: tuple[torch.Tensor, torch.Tensor],
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    """
    Computes the spectral component of AR-STU U feature vectors.

    Args:
        inputs (torch.Tensor): Input tensor of shape [bsz, sl, d_in].
        eigh (tuple[torch.Tensor, torch.Tensor]): Tuple of (eigenvalues, eigenvectors).
            eigenvalues shape: [K], eigenvectors shape: [sl, K].
        m_phi_plus (torch.Tensor): Positive spectral filter of shape [K, d_in, d_out].
        m_phi_minus (torch.Tensor): Negative spectral filter of shape [K, d_in, d_out].
        k_y (int): Number of time steps to shift.

    Returns:
        torch.Tensor: Spectral component of shape [bsz, sl, d_out].
    """
    sigma, phi = eigh
    bsz, sl, d_in = inputs.shape
    K = sigma.shape[0]
    d_out = m_phi_plus.shape[2]

    # Compute U+ and U-
    U_plus, U_minus = conv(inputs, phi)

    # Initialize result tensor
    result = torch.zeros(bsz, sl, d_out, device=inputs.device)

    # Compute the spectral component
    sigma_pow = sigma.pow(0.25)

    # Compute the spectral component
    for b in range(bsz):
        for t in range(sl):
            for k in range(K):
                for i in range(d_in):
                    for o in range(d_out):
                        # Positive component
                        if t >= k_y:
                            result[b, t, o] += (
                                m_phi_plus[k, i, o]
                                * sigma[k].pow(0.25)
                                * U_plus[b, t - k_y, k, i]
                            )

                        # Negative component
                        if t >= k_y:
                            result[b, t, o] += (
                                m_phi_minus[k, i, o]
                                * sigma_pow[k]
                                * U_minus[b, t - k_y, k, i]
                            )

    return result


def compute_spectral_opt(
    inputs: torch.Tensor,
    eigh: tuple[torch.Tensor, torch.Tensor],
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    sigma, phi = eigh
    _, K = phi.shape
    U_plus, U_minus = conv(inputs, phi)  # -> [bsz, sl, K, d_in] x 2
    U_plus_shifted = shift(U_plus, k_y)
    U_minus_shifted = shift(U_minus, k_y)

    # Add extra dim at the end
    sigma_root = sigma.pow(0.25).view(1, 1, K, 1)

    U_pluss, U_minuss = U_plus_shifted * sigma_root, U_minus_shifted * sigma_root

    spectral_plus = torch.einsum("bsKd,Kdo->bso", U_pluss, m_phi_plus)
    spectral_minus = torch.einsum("bsKd,Kdo->bso", U_minuss, m_phi_minus)

    return spectral_plus + spectral_minus


def compute_spectral_opt_stacked(
    inputs: torch.Tensor,
    eigh: tuple[torch.Tensor, torch.Tensor],
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    sigma, phi = eigh
    _, K = phi.shape
    U_plus_tilde, U_minus_tilde = conv(inputs, phi)  # -> [bsz, sl, K, d_in]
    U_shifted = torch.stack(
        [shift(U_plus_tilde, k_y), shift(U_minus_tilde, k_y)],
        dim=-1,  # -> [bsz, sl, K, d_in, 2]
    )

    sigma_root = sigma.pow(0.25).view(1, 1, K, 1, 1)
    U_weighted = U_shifted * sigma_root  # -> [bsz, sl, K, d_in, 2]

    m_phi = torch.stack([m_phi_plus, m_phi_minus], dim=-1)  # -> [K, d_in, d_out, 2]

    result = torch.einsum("bskid,kiod->bso", U_weighted, m_phi)

    return result


def generate_random_inputs(
    bsz: int, sl: int, d_in: int, d_out: int, K: int
) -> tuple[torch.Tensor, ...]:
    print(
        f"\nGenerating random inputs with bsz={bsz}, sl={sl}, d_in={d_in}, d_out={d_out}, K={K}"
    )
    inputs = torch.randn(bsz, sl, d_in, device=device)
    sigma = torch.rand(K, device=device)
    phi = torch.randn(sl, K, device=device)
    m_phi_plus = torch.randn(K, d_in, d_out, device=device)
    m_phi_minus = torch.randn(K, d_in, d_out, device=device)
    print("Generated input shapes:")
    print(f"inputs: {inputs.shape}")
    print(f"sigma: {sigma.shape}")
    print(f"phi: {phi.shape}")
    print(f"m_phi_plus: {m_phi_plus.shape}")
    print(f"m_phi_minus: {m_phi_minus.shape}")
    return inputs, (sigma, phi), m_phi_plus, m_phi_minus


def time_function(func, *args):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up run
    _ = func(*args)
    torch.cuda.synchronize()

    # Timed run
    start_event.record()
    output = func(*args)
    end_event.record()

    torch.cuda.synchronize()

    return output, start_event.elapsed_time(end_event)


def compare_outputs(
    outputs: dict[str, torch.Tensor],
    reference_output: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    eps: float = 1e-8,
) -> dict[str, tuple[bool, float, float]]:
    results = {}
    for name, output in outputs.items():
        are_equal = torch.allclose(reference_output, output, rtol=rtol, atol=atol)
        if not are_equal:
            diff = torch.abs(reference_output - output)
            max_diff = torch.max(diff).item()
            relative_diff = torch.max(diff / (torch.abs(reference_output) + eps)).item()
            results[name] = (are_equal, max_diff, relative_diff)
        else:
            results[name] = (are_equal, 0, 0)
    return results


def run_tests(
    bsz: int, sl: int, d_in: int, d_out: int, K: int, k_y: int, num_runs: int = 30
) -> tuple[dict[str, list[float]], dict[str, tuple[bool, float, float]]]:
    inputs, eigh, m_phi_plus, m_phi_minus = generate_random_inputs(
        bsz, sl, d_in, d_out, K
    )

    functions = {
        "compute_spectral": compute_spectral,
        "compute_spectral_opt": compute_spectral_opt,
        "compute_spectral_opt_stacked": compute_spectral_opt_stacked,
    }

    results = {name: [] for name in functions}
    outputs = {}

    print("\nStarting warmup...")
    for func in functions.values():
        for _ in range(5):
            _ = func(inputs, eigh, m_phi_plus, m_phi_minus, k_y)

    print("\nStarting timed runs...")
    for name, func in functions.items():
        for _ in range(num_runs):
            output, time_taken = time_function(
                func, inputs, eigh, m_phi_plus, m_phi_minus, k_y
            )
            results[name].append(time_taken)
        outputs[name] = output

    equality_results = compare_outputs(outputs, outputs["compute_spectral"])

    return results, equality_results


def format_results(
    timing_results: dict[str, list[float]],
    equality_results: dict[str, tuple[bool, float, float]],
    config: dict[str, any],
    eps: float = 1e-8,
) -> PrettyTable:
    table = PrettyTable()
    table.field_names = [
        "Function",
        "Avg Time (ms)",
        "Speedup",
        "Max Diff",
        "Max Rel Diff",
        "Equal",
    ]

    base_time = sum(timing_results["compute_spectral"]) / (
        len(timing_results["compute_spectral"]) + eps
    )

    for name, times in timing_results.items():
        avg_time = sum(times) / (len(times) + eps)
        speedup = base_time / (avg_time + eps) if name != "compute_spectral" else 1.0
        equal, max_diff, rel_diff = equality_results.get(name, (True, 0, 0))

        table.add_row(
            [
                name,
                f"{avg_time:.4f}",
                f"{speedup:.2f}x",
                f"{max_diff:.6e}",
                f"{rel_diff:.6e}",
                "Yes" if equal else "No",
            ]
        )

    table.title = f"Results for config: {config}"
    return table


def main():
    test_configs = [
        {"bsz": 1, "sl": 2, "d_in": 2, "d_out": 2, "K": 2, "k_y": 2}
        # {"bsz": 8, "sl": 10, "d_in": 4, "d_out": 4, "K": 24, "k_y": 2}
        # {"bsz": 8, "sl": 1000, "d_in": 512, "d_out": 512, "K": 24, "k_y": 2},
        # {"bsz": 16, "sl": 500, "d_in": 256, "d_out": 256, "K": 32, "k_y": 3},
    ]

    for config in test_configs:
        print(f"\nRunning test with configuration: {config}")
        timing_results, equality_results = run_tests(**config)
        table = format_results(timing_results, equality_results, config)
        print(table)


if __name__ == "__main__":
    main()
