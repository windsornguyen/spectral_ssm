import torch
import time
from typing import Tuple


def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k == 0:
        return u
    bsz, sl, K, d = u.shape
    padding = torch.zeros(bsz, k, K, d, device=u.device)
    if k < sl:
        shifted = torch.cat([padding, u[:, :-k]], dim=1)
    else:
        shifted = padding[:, :sl]
    return shifted


def compute_spectral(
    U_plus: torch.Tensor,
    U_minus: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    sigma: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    K, d_out, d_in = m_phi_plus.shape
    bsz, sl, _, _ = U_plus.shape

    # Shift U_plus and U_minus by k_y time steps
    U_plus_shifted = shift(U_plus, k_y)
    U_minus_shifted = shift(U_minus, k_y)

    # Compute σ^(1/4)
    sigma_root = (sigma**0.25).view(1, 1, K, 1)

    # Compute the spectral component
    U_plus_weighted = U_plus_shifted * sigma_root

    spectral_plus = torch.einsum("bski,kio->bso", U_plus_weighted, m_phi_plus)

    U_minus_weighted = U_minus_shifted * sigma_root
    spectral_minus = torch.einsum("bski,kio->bso", U_minus_weighted, m_phi_minus)

    result = spectral_plus + spectral_minus

    return result


def compute_spectral_opt(
    U_plus: torch.Tensor,
    U_minus: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    sigma: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    U_shifted = torch.stack([shift(U_plus, k_y), shift(U_minus, k_y)], dim=-1)

    # Compute σ^(1/4) and combine operations
    sigma_root = (sigma**0.25).view(1, 1, -1, 1, 1)
    U_weighted = U_shifted * sigma_root

    # Combine m_phi matrices
    m_phi = torch.stack([m_phi_plus, m_phi_minus], dim=-1)

    # Compute the spectral component in a single einsum
    result = torch.einsum("bskid,kiod->bso", U_weighted, m_phi)

    return result


def generate_random_inputs(
    bsz: int, sl: int, d_in: int, d_out: int, K: int
) -> Tuple[torch.Tensor, ...]:
    print(
        f"\nGenerating random inputs with bsz={bsz}, sl={sl}, d_in={d_in}, d_out={d_out}, K={K}"
    )
    U_plus = torch.randn(bsz, sl, K, d_in)
    U_minus = torch.randn(bsz, sl, K, d_in)
    m_phi_plus = torch.randn(K, d_out, d_in)
    m_phi_minus = torch.randn(K, d_out, d_in)
    sigma = torch.rand(K)
    print("Generated input shapes:")
    print(f"U_plus: {U_plus.shape}")
    print(f"U_minus: {U_minus.shape}")
    print(f"m_phi_plus: {m_phi_plus.shape}")
    print(f"m_phi_minus: {m_phi_minus.shape}")
    print(f"sigma: {sigma.shape}")
    return U_plus, U_minus, m_phi_plus, m_phi_minus, sigma


def profile_spectral(
    bsz: int, sl: int, d_in: int, d_out: int, K: int, k_y: int, num_runs: int = 30
):
    print("\nProfiling spectral computation with:")
    print(
        f"bsz={bsz}, sl={sl}, d_in={d_in}, d_out={d_out}, K={K}, k_y={k_y}, num_runs={num_runs}"
    )

    inputs = generate_random_inputs(bsz, sl, d_in, d_out, K)
    U_plus, U_minus, m_phi_plus, m_phi_minus, sigma = inputs

    print("\nStarting warmup...")
    for _ in range(5):
        compute_spectral(U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y)
        compute_spectral_opt(U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y)

    print("\nStarting timed runs...")

    # Original implementation
    start_time = time.time()
    for _ in range(num_runs):
        result = compute_spectral(U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y)
    end_time = time.time()
    avg_time_original = (end_time - start_time) / num_runs

    # Optimized implementation
    start_time = time.time()
    for _ in range(num_runs):
        result_opt = compute_spectral_opt(
            U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y
        )
    end_time = time.time()
    avg_time_opt = (end_time - start_time) / num_runs

    print("\nProfiling results:")
    print(f"Average execution time (original): {avg_time_original:.6f} seconds")
    print(f"Average execution time (optimized): {avg_time_opt:.6f} seconds")
    print(f"Speedup: {avg_time_original / avg_time_opt:.2f}x")
    print(f"Output shape: {result.shape}")

    # Check for correctness
    assert torch.allclose(
        result, result_opt, rtol=1e-4, atol=1e-4
    ), "Results do not match!"
    print("Results match between original and optimized versions.")


def compare_implementations(bsz: int, sl: int, d_in: int, d_out: int, K: int, k_y: int):
    print("\nComparing implementations with:")
    print(f"bsz={bsz}, sl={sl}, d_in={d_in}, d_out={d_out}, K={K}, k_y={k_y}")

    inputs = generate_random_inputs(bsz, sl, d_in, d_out, K)
    U_plus, U_minus, m_phi_plus, m_phi_minus, sigma = inputs

    result_original = compute_spectral(
        U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y
    )
    result_optimized = compute_spectral_opt(
        U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y
    )

    if torch.allclose(result_original, result_optimized, rtol=1e-4, atol=1e-4):
        print("Results match between original and optimized versions.")
    else:
        print("Results do not match!")
        print(
            f"Max difference: {torch.max(torch.abs(result_original - result_optimized))}"
        )


if __name__ == "__main__":
    print("--- Comparing implementations ---")
    compare_implementations(8, 1024, 32, 32, 24, 2)

    print("\n--- Profiling with various input sizes ---")
    profile_spectral(8, 1024, 32, 32, 24, 2)
