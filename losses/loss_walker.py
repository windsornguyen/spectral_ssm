# ==============================================================================#
# Authors: Isabel Liu
# File: loss_walker.py
# ==============================================================================#

"""Customized Loss for Walker2D-v1 Task."""

import torch
import torch.nn as nn


class Walker2DLoss(nn.Module):
    def __init__(self):
        super(Walker2DLoss, self).__init__()

    def forward(
        self,
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
            """
            Compute the loss and metrics for a batch of data.

            Args:
                outputs (torch.Tensor): The model outputs.
                targets (torch.Tensor): The target labels.

            Returns:
                tuple[torch.Tensor, Dict[str, float]]: 
                A tuple of the loss and a Dictionary of metrics.
            """
            total_loss = torch.tensor(0.0, device=outputs.device)
            coordinate_loss = torch.tensor(0.0, device=outputs.device)
            angle_loss = torch.tensor(0.0, device=outputs.device)
            coordinate_velocity_loss = torch.tensor(0.0, device=outputs.device)
            angular_velocity_loss = torch.tensor(0.0, device=outputs.device)

            for i in range(outputs.shape[1]):
                loss = (outputs[:, i] - targets[:, i]) ** 2

                # TODO: Write a more sophisticated loss function?
                if i in (0, 1):  # coordinates of the front tip
                    loss /= 2.5
                    coordinate_loss += loss.mean()
                elif i in (2, 3, 4, 5, 6, 7, 8):  # angles of the front tip and limbs
                    loss /= 0.5
                    angle_loss += loss.mean()
                elif i in (9, 10):  # coordinate velocities of the front tip
                    loss /= 2
                    coordinate_velocity_loss += loss.mean()
                elif i in (11, 12, 13, 14, 15, 16, 17):  # angular velocities of the front tip and limbs
                    loss /= 2.5
                    angular_velocity_loss += loss.mean()

                total_loss += loss.mean()

            total_loss /= outputs.shape[1]
            coordinate_loss /= 2
            orientation_loss = 'N/A for Walker2D-v1'
            angle_loss /= 7
            coordinate_velocity_loss /= 2
            angular_velocity_loss /= 7
            metrics = {
                'coordinate_loss': coordinate_loss.item(), 
                'orientation_loss': orientation_loss,
                'angle_loss': angle_loss.item(), 
                'coordinate_velocity_loss': coordinate_velocity_loss.item(), 
                'angular_velocity_loss': angular_velocity_loss.item()
            }
            return total_loss, metrics
