# =============================================================================#
# Authors: Windsor Nguyen
# File: rms_norm.py
# =============================================================================#

# TODO: Move this to another directory that makes sense.
# TODO: In general, organize the utils directory better.

"""
Implementation of Root Mean Square Layer Normalization (RMSNorm).

RMSNorm is a simplified version of Layer Normalization that only
performs scaling, leading to improved performance and stability.
"""

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): The dimension of the input tensor to be normalized.
        eps (float, optional): A small value for numerical stability. Default: 1e-6
        elementwise_affine (bool, optional): If True, learns an affine transform. Default: True

    Shape:
        - Input: (*, dim)
        - Output: (*, dim)
    
    Reference:
        "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019),
        https://arxiv.org/abs/1910.07467.
    """

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dim, f"Expected last dimension {self.dim}, but got {x.shape[-1]}"
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            output = output * self.weight
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"
