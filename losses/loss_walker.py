# ==============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: loss_walker.py
# ==============================================================================#

"""Customized Loss for Walker2D-v1 Task."""

import torch
import torch.nn as nn


class Walker2DLoss(nn.Module):
    def __init__(self):
        super(Walker2DLoss, self).__init__()
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
        feature_groups = {
            "coordinate_loss": (0, 1),
            "angle_loss": (2, 3, 4, 5, 6, 7, 8),
            "coordinate_velocity_loss": (9, 10),
            "angular_velocity_loss": (11, 12, 13, 14, 15, 16, 17),
        }

        metrics = {
            metric_name: self.mse_loss(
                outputs[:, :, indices], targets[:, :, indices]
            ).item()
            for metric_name, indices in feature_groups.items()
        }

        # N/A for orientation_loss as it's not applicable for Walker2D
        metrics["orientation_loss"] = "N/A for Walker2D-v1"

        return total_loss, metrics
