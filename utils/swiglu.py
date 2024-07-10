# =============================================================================#
# Authors: Windsor Nguyen
# File: swiglu.py
# =============================================================================#

# TODO: Move this to another directory that makes sense.
# TODO: In general, organize the utils directory better.

"""
The SwiGLU activation function,
from "GLU Variants Improve Transformer" (Shazeer, 2020).

From the paper:
'We offer no explanation as to why these architectures seem to work;
we attribute their success, as all else, to __divine benevolence__.'
"""

import torch.nn as nn


class SwiGLU(nn.Module):
    """
    The SwiGLU activation function as proposed by Noam Shazeer.

    This module implements the SwiGLU function defined as:
    FFN_SwiGLU(x, W, V, W2) = (Swish_{1}(xW) ⊙ (xV))W2
    where ⊙ denotes the Hadamard product and Swish_{1} is the Swish function with β=1.

    Note: The Swish function with β=1 is equivalent to PyTorch's SiLU function.

    Args:
        dim (int): Input and output dimension.
        h_dim (int): Hidden dimension.
        bias (bool, optional): If false, additive biases will not be learned.
    
    Reference:
        "GLU Variants Improve Transformer" (Shazeer, 2020),
        https://arxiv.org/abs/2002.05202.
    """

    def __init__(self, dim, h_dim, bias=False):
        super().__init__()
        self.w = nn.Linear(dim, h_dim, bias=bias)
        self.v = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.w2(self.swish(self.w(x)) * self.v(x))
