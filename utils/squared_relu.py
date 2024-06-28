# =============================================================================#
# Authors: Windsor Nguyen
# File: squared_relu.py
# =============================================================================#

# TODO: Move this to another directory that makes sense.
# TODO: In general, organize the utils directory better.

import torch
import torch.nn.functional as F
import torch.nn as nn


class SquaredReLU(nn.Module):
    """
    The SquaredReLU activation function as proposed in the Primer paper,
    "Primer: Searching for Efficient Transformers for Language Modeling"
    by So et al. (2021).

    This activation function is defined as:
    SquaredReLU(x) = (max(0, x))^2

    It was found to improve the efficiency of Transformers in language modeling.
    """

    def forward(self, x):
        return torch.square(F.relu(x))
