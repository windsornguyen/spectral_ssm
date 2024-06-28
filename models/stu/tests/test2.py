import triton
import triton.language as tl
import torch

# Define a Triton kernel for batched matrix multiplication
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_za, stride_ma, stride_na,
    stride_zb, stride_nb, stride_kb,
    stride_zc, stride_mc, stride_nc,
    Z, M, N, K,
    BLOCK_SIZE: tl.constexpr
):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_k = tl.arange(0, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    A_ptrs = A_ptr + pid_z * stride_za + offs_m[:, None] * stride_ma + offs_k[None, :] * stride_na
    B_ptrs = B_ptr + pid_z * stride_zb + offs_k[:, None] * stride_kb + offs_n[None, :] * stride_nb
    C_ptrs = C_ptr + pid_z * stride_zc + offs_m[:, None] * stride_mc + offs_n[None, :] * stride_nc
    
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float64)
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(A_ptrs + k)
        b = tl.load(B_ptrs + k)
        accumulator += tl.dot(a, b)
        
    tl.store(C_ptrs, accumulator)

def batched_matmul(A, B, C, Z, M, N, K):
    BLOCK_SIZE = 16
    grid = (Z, (M + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    batched_matmul_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        Z, M, N, K,
        BLOCK_SIZE
    )

# Example usage
A = torch.randn((4, 29, 58)).cuda()
B = torch.randn((4, 58, 58)).cuda()
C = torch.empty((4, 29, 58)).cuda()

batched_matmul(A, B, C, 4, 29, 58, 58)

print(C)
