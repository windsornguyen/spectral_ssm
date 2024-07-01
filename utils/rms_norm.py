# =============================================================================#
# Authors: Windsor Nguyen
# File: rms_norm.py
# =============================================================================#

# TODO: Move this to another directory that makes sense.
# TODO: In general, organize the utils directory better.

"""
Implementation of Root Mean Square Layer Normalization (RMSNorm).

Based on the paper:
"Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
https://arxiv.org/abs/1910.07467

RMSNorm is a simplified version of Layer Normalization that only
performs scaling, leading to improved performance and stability.
"""

import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.wt = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute root mean square
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
    
        # Normalize and scale
        x_norm = x / rms

        return self.wt * x_norm
