# ==============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: loss_cartpole.py
# ==============================================================================#

"""Customized Loss for Cartpole-v1 Task."""

import torch
import torch.nn as nn


class CartpoleLoss(nn.Module):
    def __init__(self):
        super(CartpoleLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the MSE loss and custom metrics for a batch of data.

        Args:
            outputs (torch.Tensor): The model outputs of shape (batch_size, seq_len, d_xt)
            targets (torch.Tensor): The target labels of shape (batch_size, seq_len, d_xt)

        Returns:
            tuple[torch.Tensor, Dict[str, float]]:
            A tuple of the MSE loss and a dictionary of custom metrics.
        """
        # Compute overall MSE loss
        total_loss = self.mse_loss(outputs, targets)

        # Compute custom metrics
        metrics = {}
        feature_groups = {
            "cart_pos_loss": (0,),
            "cart_vel_loss": (1,),
            "pole_ang_loss": (2,),
            "pole_angvel_loss": (3,),
        }

        for metric_name, feature_indices in feature_groups.items():
            metric_loss = self.mse_loss(
                outputs[:, :, feature_indices], targets[:, :, feature_indices]
            )
            metrics[metric_name] = metric_loss.item()

        metrics["loss"] = total_loss.item()

        return total_loss, metrics
