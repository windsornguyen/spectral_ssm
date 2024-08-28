import torch
from flashfftconv import FlashFFTConv

def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    indices = torch.arange(1, seq_len + 1)
    hankel = indices[:, None] + indices[None, :]

    if use_hankel_L:
        sgn = -(1.0 ** (hankel - 2.0)) + 1.0
        denom = (hankel + 3.0) * (hankel - 1.0) * (hankel + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (hankel**3 - hankel)

    return Z

def get_spectral_filters(
    seq_len: int, K: int, use_hankel_L: bool = False,
) -> torch.Tensor:
    assert torch.cuda.is_available(), "CUDA is required."
    device = torch.device("cuda")
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma, phi = sigma[-K:], phi[:, -K:]
    phi *= sigma
    return phi

def preconvolve(phi: torch.Tensor, n: int, approx: bool = True) -> tuple[torch.Tensor, int]:
    seq_len, K = phi.shape
    phi = phi.view(1, seq_len, K, 1)
    signal = torch.fft.rfft(phi, n=n, dim=1)
    return signal

def convolve(u: torch.Tensor, v: torch.Tensor, n: int, approx: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1
    if approx:
        _, d_out = v.shape
        v = v.view(1, seq_len, d_out, 1).to(torch.float32)
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, seq_len, K, 1, 1).to(torch.float32)
        u = u.view(bsz, seq_len, 1, d_in).expand(bsz, seq_len, K, d_in)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus, U_minus

def flash_convolve(
    u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, approx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    sgn = torch.full((1, 1, seq_len), 1, device=u.device)
    sgn[:, :, 1::2] = -1

    if approx:
        u = u.to(torch.bfloat16).transpose(1, 2).contiguous()
        v = v.to(torch.float32).transpose(0, 1).contiguous()
        u_conv = torch.stack([u, u * sgn], dim=0).reshape(2 * bsz, d_in, seq_len)
    else:
        u_k = u.to(torch.bfloat16).transpose(1, 2).repeat_interleave(K, dim=1).contiguous()
        v = v.to(torch.float32).transpose(0, 1).repeat(d_in, 1).contiguous()
        u_conv = torch.stack([u_k, u_k * sgn], dim=0).reshape(2 * bsz, K * d_in, seq_len)

    U_conv = flash_fft(u_conv, v)
    u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

    if approx:
        u_minus = u_minus * sgn
        U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
    else:
        sgn = sgn.unsqueeze(-1).transpose(1, 2)
        U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

    return U_plus, U_minus

# Additional functions
def shift(u: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k == 0:
        return u
    shifted = torch.roll(u, shifts=k, dims=1)
    shifted[:, :k] = 0
    return shifted

def compute_ar_u(M_u: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
    k_u = M_u.shape[0]
    u_shifted = torch.stack([shift(u_t, i) for i in range(k_u)], dim=1)
    ar_u = torch.einsum("bksi,koi->bso", u_shifted, M_u)
    return ar_u

def compute_ar_y(M_y: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    d_out, k_y, _ = M_y.shape
    bsz, sl, _ = y_t.shape

    eye = torch.eye((k_y - 1) * d_out, k_y * d_out, dtype=y_t.dtype, device=y_t.device)
    A = M_y.reshape(d_out, k_y * d_out)
    A = torch.cat([A, eye], dim=0)
    A = A.unsqueeze(0).expand(bsz, k_y * d_out, k_y * d_out)

    padding = torch.zeros(bsz, sl, (k_y - 1) * d_out, dtype=y_t.dtype, device=y_t.device)
    state = torch.cat([y_t, padding], dim=2)
    state = state.view(bsz, sl, k_y * d_out, 1)

    y = state[:, 0]
    ys = [y[:, :d_out, 0]]

    for i in range(1, sl):
        y_next = state[:, i]
        y = torch.bmm(A, y) + y_next
        ys.append(y[:, :d_out, 0])

    return torch.stack(ys, dim=1)
