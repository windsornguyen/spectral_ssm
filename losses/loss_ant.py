# ==============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: loss_ant.py
# ==============================================================================#

"""Customized Loss for Ant-v1 Task."""

import torch
import torch.nn as nn


class AntLoss(nn.Module):
    def __init__(self):
        super(AntLoss, self).__init__()
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
            "coordinate_loss": (0, 1, 2),
            "orientation_loss": (3, 4, 5, 6),
            "angle_loss": (7, 8, 9, 10, 11, 12, 13, 14),
            "coordinate_velocity_loss": (15, 16, 17, 18, 19, 20),
            "angular_velocity_loss": (21, 22, 23, 24, 25, 26, 27, 28),
        }

        for metric_name, feature_indices in feature_groups.items():
            metric_loss = self.mse_loss(
                outputs[:, :, feature_indices], targets[:, :, feature_indices]
            )
            metrics[metric_name] = metric_loss.item()

        return total_loss, metrics
