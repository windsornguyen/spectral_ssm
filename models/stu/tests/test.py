import torch
import time
import numpy as np

# Define constants
D_OUT = 29
K = 3
BATCH_SIZE = 5
SEQ_LEN = 100
ATOL = 1e-4
N_RUNS = 30

# Generate random input tensors with double precision
torch.random.manual_seed(1337)
np.random.seed(1337)

m_y = torch.randn(D_OUT, K, D_OUT, dtype=torch.float64).cuda()
deltas = torch.randn(BATCH_SIZE, SEQ_LEN, D_OUT, dtype=torch.float64).cuda()

# Define function 1
@torch.jit.script
def og(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    d_out, k, _ = m_y.shape
    bsz, seq_len, _ = deltas.shape
    carry = torch.zeros(
        (bsz, d_out, k), device=deltas.device, dtype=deltas.dtype
    )
    ys = torch.zeros(
        (bsz, seq_len, d_out), device=deltas.device, dtype=deltas.dtype
    )

    for i in range(seq_len):
        output = torch.einsum('ijk,bkj->bi', m_y, carry) + deltas[:, i, :]
        ys[:, i, :] = output
        carry = torch.roll(carry, shifts=1, dims=2)
        carry[:, :, 0] = output

    return ys


# Define function 2
@torch.jit.script
def bmm(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    d_out, k, _ = m_y.shape
    bsz, sl, _ = deltas.shape

    # Define the transition matrix A, and add bsz for bmm
    A = m_y.view(d_out, -1)  # Reshape m_y to [d_out, k * d_out] for concat
    eye = torch.eye((k - 1) * d_out, k * d_out, dtype=deltas.dtype, device=deltas.device)
    A = torch.cat([A, eye], dim=0)
    A = A.unsqueeze(0).expand(bsz, -1, -1) # -> [bsz, k * d_out, k * d_out]

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
    for i in range(1, sl):
        y = torch.bmm(A, y) + carry[:, i]
        ys.append(y[:, :d_out, 0])
    ys = torch.stack(ys, dim=1) # -> [bsz, sl, d_out]
    
    return ys


# Define function 3
@torch.jit.script
def evans(m_y: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    d_out, k, _ = m_y.shape
    bsz, seq_len, _ = deltas.shape

    device = m_y.device

    A = torch.cat(
        [
            m_y.reshape(d_out, k * d_out).to(device),
            torch.eye(
                (k - 1) * d_out, k * d_out, device=device, dtype=deltas.dtype
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
                dtype=deltas.dtype,
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


def measure_time(func, m_y, deltas, n_runs=N_RUNS):
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        func(m_y, deltas)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times), np.std(times)


mean1, std1 = measure_time(og, m_y, deltas)
mean2, std2 = measure_time(bmm, m_y, deltas)
mean3, std3 = measure_time(evans, m_y, deltas)

print(f'Function 1 (og) - Mean: {mean1:.6f}, Stddev: {std1:.6f}')
print(f'Function 2 (bmm) - Mean: {mean2:.6f}, Stddev: {std2:.6f}')
print(f'Function 3 (evans) - Mean: {mean3:.6f}, Stddev: {std3:.6f}')

# Ensure results match
output1 = og(m_y, deltas)
output2 = bmm(m_y, deltas)
output3 = evans(m_y, deltas)


def compare_outputs(output1, output2, output3):
    mismatch_count = 0
    for i in range(output1.shape[0]):
        for j in range(output1.shape[1]):
            for k in range(output1.shape[2]):
                if not torch.allclose(
                    output1[i, j, k], output2[i, j, k], atol=ATOL
                ):
                    print(f'Mismatch at {i}, {j}, {k} for og and bmm:')
                    print(
                        f'og: {output1[i, j, k]}, bmm: {output2[i, j, k]}, diff: {output1[i, j, k] - output2[i, j, k]}'
                    )
                    mismatch_count += 1
                if not torch.allclose(
                    output1[i, j, k], output3[i, j, k], atol=ATOL
                ):
                    print(f'Mismatch at {i}, {j}, {k} for og and evans:')
                    print(
                        f'og: {output1[i, j, k]}, evans: {output3[i, j, k]}, diff: {output1[i, j, k] - output3[i, j, k]}'
                    )
                    mismatch_count += 1
                if not torch.allclose(
                    output2[i, j, k], output3[i, j, k], atol=ATOL
                ):
                    print(f'Mismatch at {i}, {j}, {k} for bmm and evans:')
                    print(
                        f'bmm: {output2[i, j, k]}, evans: {output3[i, j, k]}, diff: {output2[i, j, k] - output3[i, j, k]}'
                    )
                    mismatch_count += 1
                if mismatch_count >= 5:
                    return


compare_outputs(output1, output2, output3)

assert torch.allclose(
    output1, output2, atol=ATOL
), 'Output mismatch between function 1 (og) and function 2 (bmm)'
assert torch.allclose(
    output1, output3, atol=ATOL
), 'Output mismatch between function 1 (og) and function 3 (evans)'
assert torch.allclose(
    output2, output3, atol=ATOL
), 'Output mismatch between function 2 (bmm) and function 3 (evans)'
