import torch

def tril_toeplitz(y: torch.Tensor) -> torch.Tensor:
    """
    Efficiently construct lower triangular Toeplitz matrices for each batch and feature.
    
    Args:
    y (torch.Tensor): Input tensor of shape (bsz, sl, d_out)
    
    Returns:
    torch.Tensor: Lower triangular Toeplitz matrices of shape (bsz, sl, sl, d_out)
    """
    batch_size, seq_len, d_out = y.shape
    
    # Create indices for the Toeplitz structure
    row_indices = torch.arange(seq_len, device=y.device)
    col_indices = torch.arange(seq_len, device=y.device)
    indices = col_indices - row_indices.unsqueeze(1)
    
    # Create the mask for lower triangular structure
    mask = indices.le(0).unsqueeze(0).unsqueeze(-1)
    
    # Expand y to match the output shape
    y_expanded = y.unsqueeze(2).expand(batch_size, seq_len, seq_len, d_out)
    
    # Shift the values along the sequence dimension
    shifted = y_expanded.gather(1, (-indices).clamp(min=0).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, seq_len, d_out))
    
    # Apply the mask to get the final result
    result = shifted * mask.to(y.dtype)
    
    return result

# Test the function
batch_size, seq_len, d_out = 2, 5, 3
y = torch.arange(1, batch_size*seq_len*d_out + 1, dtype=torch.float32).reshape(batch_size, seq_len, d_out)
result = tril_toeplitz(y)
print(f"Input shape: {y.shape}")
print(f"Output shape: {result.shape}")
print("\nInput tensor:")
print(y)
print("\nOutput tensor (first batch, first feature):")
print(result[0, :, :, 0])
