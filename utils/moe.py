# =============================================================================#
# Authors: Windsor Nguyen
# File: moe.py
# =============================================================================#

# TODO: Move this to another directory that makes sense.
# TODO: In general, organize the utils directory better.

"""
Simple Mixture-of-Experts architecture from 
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
        num_experts_per_timestep (int): Number of experts to use per token.
    """
    def __init__(self, configs, experts: list[nn.Module], gate: nn.Module):
        super(MoE, self).__init__()
        assert len(experts) > 0, "MoE requires at least one expert"
        assert configs.num_experts_per_timestep <= configs.num_experts, "num_experts_per_timestep must be <= num_experts"
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = configs.num_experts
        self.num_experts_per_timestep = configs.num_experts_per_timestep

    def forward(self, inputs: torch.Tensor):
        # Fuse inputs to process all tokens across all sequences in batch at once
        inputs_fused = inputs.view(-1, inputs.shape[-1]) # (bsz * sl, n_embd)
        
        # Compute logits for each expert
        gate_logits = self.gate(inputs_fused) # (bsz * sl, num_experts)

        # Select the top num_experts_per_timestep experts and their _raw_ logits
        weights, selected_experts = torch.topk(
            gate_logits,
            self.num_experts_per_timestep
        ) # Both: (bsz * sl, num_experts_per_timestep)

        # Normalize the logits to get the "probabilities"
        weights = F.softmax(weights, dim=-1, dtype=torch.float).type_as(inputs)

        # Allocate tensor for final output of the MoE layer
        results = torch.zeros_like(inputs_fused)

        # For each expert, determine which tokens are selected for that expert
        for idx, expert in enumerate(self.experts):
            # Find all positions where current expert was selected as a top expert
            batch_idx, nth_expert = torch.where(selected_experts == idx)

            # Select inputs and weights for current expert
            expert_inputs = inputs_fused[batch_idx]
            expert_weights = weights[batch_idx, nth_expert, None]
            
            # Apply expert to selected inputs
            expert_outputs = expert(expert_inputs)
            
            # Weight the expert output and add to results
            weighted_output = expert_weights * expert_outputs
            results[batch_idx] += weighted_output
        
        # Reshape results back to original shape
        return results.view_as(inputs)
