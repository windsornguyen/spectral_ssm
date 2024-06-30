import torch
import torch.nn as nn
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def compute_m_phi_plus(
    U_plus: torch.Tensor, m_phi_plus: torch.Tensor, sigma: torch.Tensor, k_y: int
) -> torch.Tensor:
    bsz, sl, K, d_in = U_plus.shape
    K, d_out, _ = m_phi_plus.shape

    sigma_root = sigma.pow(0.25).view(1, 1, K, 1)
    U_plus_shifted = shift(U_plus, k_y)
    U_plus_weighted = U_plus_shifted * sigma_root
    result = torch.einsum("bski,kio->bso", U_plus_weighted, m_phi_plus)

    return result


def compute_m_phi_minus(
    U_minus: torch.Tensor, m_phi_minus: torch.Tensor, sigma: torch.Tensor, k_y: int
) -> torch.Tensor:
    return compute_m_phi_plus(U_minus, m_phi_minus, sigma, k_y)


def nearest_power_of_2(x: int) -> int:
    return 2 ** (x - 1).bit_length()


def conv(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    bsz, sl, K, d_in = u.shape
    k = v.shape[1]
    n = nearest_power_of_2(sl * 2 - 1)
    v = v.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, d_in)
    u = u.unsqueeze(2).expand(-1, -1, k, -1)
    V = torch.fft.rfft(v, n=n, dim=1)
    U = torch.fft.rfft(u, n=n, dim=1)
    Z = V * U
    z = torch.fft.irfft(Z, n=n, dim=1)
    return z[:, :sl]


def compute_y_t(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    d_out, k, _ = m_y.shape
    bsz, sl, _ = deltas.shape

    A = m_y.view(d_out, k * d_out)
    eye = torch.eye(
        (k - 1) * d_out, k * d_out, dtype=deltas.dtype, device=deltas.device
    )
    A = torch.cat([A, eye], dim=0)
    A = A.unsqueeze(0).expand(bsz, k * d_out, k * d_out)

    padding = torch.zeros(
        bsz, sl, (k - 1) * d_out, dtype=deltas.dtype, device=deltas.device
    )
    carry = torch.cat([deltas, padding], dim=2)
    carry = carry.view(bsz, sl, k * d_out, 1)

    y = carry[:, 0]
    ys = [y[:, :d_out, 0]]

    for i in range(1, sl):
        y = torch.bmm(A, y) + carry[:, i]
        ys.append(y[:, :d_out, 0])
    ys = torch.stack(ys, dim=1)

    return ys


def compute_ar_x_preds(m_u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    bsz, sl, d_in = x.shape
    k_u, d_out, _ = m_u.shape

    o = torch.einsum("koi,bli->bklo", m_u, x)
    rolled_o = torch.stack(
        [torch.roll(o[:, i], shifts=i, dims=1) for i in range(k_u)], dim=1
    )
    mask = torch.triu(torch.ones((k_u, sl), device=m_u.device)).view(k_u, sl, 1)
    return torch.sum(rolled_o * mask, dim=1)


def compute_x_tilde(
    inputs: torch.Tensor, eigh: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    eig_vals, eig_vecs = eigh
    k = eig_vals.size(0)
    bsz, sl, K, d_in = inputs.shape

    x_spectral = conv(eig_vecs, inputs)
    eig_vals = eig_vals.view(1, 1, k, 1)
    x_tilde = x_spectral * eig_vals.pow(0.25)
    return x_tilde.view(bsz, sl, k * d_in)


def compute_spectral_v1(
    U_plus: torch.Tensor,
    U_minus: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    sigma: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    return compute_m_phi_plus(U_plus, m_phi_plus, sigma, k_y) + compute_m_phi_minus(
        U_minus, m_phi_minus, sigma, k_y
    )


def compute_spectral_v2(
    U_plus: torch.Tensor,
    U_minus: torch.Tensor,
    m_phi_plus: torch.Tensor,
    m_phi_minus: torch.Tensor,
    sigma: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    K, d_out, d_in = m_phi_plus.shape
    bsz, sl, _, _ = U_plus.shape

    U_plus_shifted = shift(U_plus, k_y)
    U_minus_shifted = shift(U_minus, k_y)
    sigma_root = sigma.pow(0.25).view(1, 1, K, 1)

    U_plus_weighted = U_plus_shifted * sigma_root
    spectral_plus = torch.einsum("bski,kio->bso", U_plus_weighted, m_phi_plus)

    U_minus_weighted = U_minus_shifted * sigma_root
    spectral_minus = torch.einsum("bski,kio->bso", U_minus_weighted, m_phi_minus)

    return spectral_plus + spectral_minus


def compute_spectral_v3(
    inputs: torch.Tensor,
    m_u: torch.Tensor,
    m_phi: torch.Tensor,
    m_y: torch.Tensor,
    eig_vals: torch.Tensor,
    eig_vecs: torch.Tensor,
    k_y: int,
) -> torch.Tensor:
    x_tilde = compute_x_tilde(inputs, (eig_vals, eig_vecs))
    delta_phi = x_tilde @ m_phi
    delta_ar_u = compute_ar_x_preds(m_u, inputs)
    y_t = compute_y_t(m_y, delta_phi + delta_ar_u)
    return y_t


def main():
    # Initialize the variables with toy values
    bsz, sl, d_in, d_out = 1, 2, 2, 2
    K = 1
    k_u = 1
    k_y = 1

    U_plus = torch.tensor(
        [[[[0.1, 0.2]], [[0.3, 0.4]]]], dtype=torch.float32, device=device
    )
    U_minus = torch.tensor(
        [[[[0.5, 0.6]], [[0.7, 0.8]]]], dtype=torch.float32, device=device
    )
    m_phi_plus = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32, device=device)
    m_phi_minus = torch.tensor([[[5, 6], [7, 8]]], dtype=torch.float32, device=device)
    sigma = torch.tensor([0.5], dtype=torch.float32, device=device)

    inputs = torch.cat([U_plus, U_minus], dim=2).squeeze(2).to(device)

    # Compute results for v1 and v2
    result_v1 = compute_spectral_v1(
        U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y
    )
    result_v2 = compute_spectral_v2(
        U_plus, U_minus, m_phi_plus, m_phi_minus, sigma, k_y
    )

    # Initialize parameters for v3 with correct dimensions
    m_u_v3 = torch.randn(k_u, d_out, d_in, device=device)
    m_phi_v3 = torch.randn(d_out * K, d_out, device=device)
    m_y_v3 = torch.randn(d_out, k_y, d_out, device=device)
    eig_vals_v3 = torch.rand(K, device=device)
    eig_vecs_v3 = torch.rand(sl, K, device=device)

    # Compute result using v3
    result_v3 = compute_spectral_v3(
        inputs, m_u_v3, m_phi_v3, m_y_v3, eig_vals_v3, eig_vecs_v3, k_y
    )

    # Compare the results
    print("Result from V1:", result_v1)
    print("Result from V2:", result_v2)
    print("Result from V3:", result_v3)
    print(
        "Are the results close? V1 and V2:",
        torch.allclose(result_v1, result_v2, atol=1e-4),
    )
    print(
        "Are the results close? V3 and V1:",
        torch.allclose(result_v3, result_v1, atol=1e-4),
    )
    print(
        "Are the results close? V3 and V2:",
        torch.allclose(result_v3, result_v2, atol=1e-4),
    )


if __name__ == "__main__":
    main()
