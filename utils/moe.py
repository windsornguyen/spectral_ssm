# =============================================================================#
# Authors: Windsor Nguyen
# File: moe.py
# =============================================================================#

# TODO: Move this to another directory that makes sense.
# TODO: In general, organize the utils directory better.

"""
The Mixture-of-Experts architecture from 
"Mixtral of Experts" by Jiang et al. (2024).
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) layer for the Transformer.

    This layer implements a Mixture of Experts approach, where multiple expert networks
    are used in parallel, with a gating mechanism to determine which experts to use
    for each input time step.

    Args:
        configs (TransformerConfigs): Configuration object containing MoE-related parameters.
        experts (list[nn.Module]): List of expert networks.
        gate (nn.Module): Gating network to determine expert selection.

    Attributes:
        experts (nn.ModuleList): List of expert networks.
        gate (nn.Module): Gating network.
        num_experts (int): Total number of experts.
        num_experts_per_tok (int): Number of experts to use per token.
    """
    def __init__(self, configs, experts: list[nn.Module], gate: nn.Module):
        super(MoE, self).__init__()
        assert len(experts) > 0, "MoE requires at least one expert"
        assert configs.num_experts_per_tok <= configs.num_experts, "num_experts_per_tok must be <= num_experts"
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = configs.num_experts
        self.num_experts_per_tok = configs.num_experts_per_tok

    def forward(self, inputs: torch.Tensor):
        inputs_fused = inputs.view(-1, inputs.shape[-1]) # (bsz * sl, n_embd)
        gate_logits = self.gate(inputs_fused)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).type_as(inputs)

        results = torch.zeros_like(inputs_fused)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_fused[batch_idx]
            )
        return results.view_as(inputs)
