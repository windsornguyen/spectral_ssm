# =============================================================================#
# Authors: Windsor Nguyen
# File: miglu.py
# =============================================================================#

"""
The MiGLU activation function,
inspired by "GLU Variants Improve Transformer" (Shazeer, 2020),
but using Mish activation instead of Swish.

Mish: A Self Regularized Non-Monotonic Neural Activation Function (Misra, 2019)
"""

import torch.nn as nn

class MiGLU(nn.Module):
    """
    The MishGLU activation function, a variant of SwiGLU using Mish activation.

    This module implements the MishGLU function defined as:
    FFN_MishGLU(x, W, V, W2) = (Mish(xW) ⊙ (xV))W2
    where ⊙ denotes the Hadamard product and Mish is PyTorch's nn.Mish function.

    Args:
        dim (int): Input and output dimension.
        h_dim (int): Hidden dimension.
        bias (bool, optional): If false, additive biases will not be learned.
    """

    def __init__(self, dim, h_dim, bias=False):
        super().__init__()
        self.w = nn.Linear(dim, h_dim, bias=bias)
        self.v = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.mish = nn.Mish()

    def forward(self, x):
        return self.w2(self.mish(self.w(x)) * self.v(x))
