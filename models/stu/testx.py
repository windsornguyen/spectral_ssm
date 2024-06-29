import torch
import time

torch.manual_seed(1337)

def compute_yt_base(M_y, y):
    bsz, seq_len, d_out = y.shape
    k_y = M_y.shape[0]
    yt = torch.zeros_like(y)
    for t in range(seq_len):
        for j in range(1, min(k_y, t) + 1):
            yt[:, t, :] += torch.matmul(y[:, t - j, :], M_y[j - 1].T)
    return yt


def compute_yt_vectorized_1(M_y, y):
    bsz, seq_len, d_out = y.shape
    k_y = M_y.shape[0]
    yt = torch.zeros_like(y)
    for j in range(1, k_y + 1):
        yt[:, j:, :] += torch.matmul(y[:, :-j, :], M_y[j - 1].T)
    return yt


def compute_yt_vectorized_2(M_y, y):
    bsz, seq_len, d_out = y.shape
    k_y = M_y.shape[0]
    o = torch.einsum("koi,bli->bklo", M_y, y)
    rolled_o = torch.stack(
        [torch.roll(o[:, i], shifts=i + 1, dims=1) for i in range(k_y)], dim=1
    )
    mask = (
        torch.tril(torch.ones((seq_len, k_y), device=y.device), diagonal=-1)
        .transpose(0, 1)
        .unsqueeze(-1)
    )
    return torch.sum(rolled_o * mask, dim=1)


def compute_yt_fully_vectorized(M_y, y):
    # bsz, seq_len, d_out = y.shape
    # k_y = M_y.shape[0]
    # o = torch.bmm(
    #     y.unsqueeze(1).expand(-1, k_y, -1, -1).reshape(-1, seq_len, d_out),
    #     M_y.transpose(-1, -2).repeat(bsz, 1, 1),
    # ).view(bsz, k_y, seq_len, d_out)
    # yt = torch.zeros_like(y)
    # for k in range(k_y):
    #     yt[:, k + 1 :] += o[:, k, : seq_len - k - 1]
    # return yt
    
    ####
    
    bsz, sl, d_out = y.shape
    k_y = M_y.shape[0]

    # Add k_y dim to y
    expanded_y = y.unsqueeze(1).expand(bsz, k_y, sl, d_out)

    # Reshape for bmm
    reshaped_y = expanded_y.reshape(bsz * k_y, sl, d_out)

    # Repeating and transposing M_y for batch matrix multiplication
    repeated_M_y = M_y.transpose(-1, -2).repeat(bsz, 1, 1)
   
    # Batch matrix multiplication
    o = torch.bmm(reshaped_y, repeated_M_y)
    o = o.view(bsz, k_y, sl, d_out)

    # Initialize yt
    yt = torch.zeros_like(y)

    # Update yt using results from o
    for k in range(k_y):
        yt[:, k + 1 :] += o[:, k, : sl - k - 1]

    return yt
    
    ####


def benchmark(func, M_y, y, name):
    torch.cuda.synchronize()
    start = time.time()
    result = func(M_y, y)
    torch.cuda.synchronize()
    end = time.time()
    return result, end - start


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bsz, seq_len, d_out, k_y = 8, 1000, 29, 3
    M_y = torch.randn(k_y, d_out, d_out, device=device)
    y = torch.randn(bsz, seq_len, d_out, device=device)

    functions = [
        ("Base", compute_yt_base),
        ("Vectorized 1", compute_yt_vectorized_1),
        ("Vectorized 2", compute_yt_vectorized_2),
        ("Fully Vectorized", compute_yt_fully_vectorized),
    ]

    results = {}
    for name, func in functions:
        torch.cuda.empty_cache()
        result, time_taken = benchmark(func, M_y, y, name)
        results[name] = (result, time_taken)
        print(f"{name} implementation time: {time_taken:.6f} seconds")

    base_time = results["Base"][1]
    for name, (result, time_taken) in results.items():
        if name != "Base":
            speedup = base_time / time_taken
            print(f"{name} vs Base speedup: {speedup:.2f}x")
            is_close = torch.allclose(results["Base"][0], result, atol=1e-6)
            print(f"{name} output matches Base: {is_close}")


if __name__ == "__main__":
    main()
