# ==============================================================================#
# Authors: Isabel Liu
# File: loss_ant.py
# ==============================================================================#

"""Customized Loss for Ant-v1 Task."""

import torch
import torch.nn as nn


class AntLoss(nn.Module):
    def __init__(self):
        super(AntLoss, self).__init__()

    def forward(
        self, 
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the loss and metrics for a batch of data.

        Args:
            outputs (torch.Tensor): The model outputs of shape (batch_size, seq_len, d_xt)
            targets (torch.Tensor): The target labels of shape (batch_size, seq_len, d_xt)

        Returns:
            tuple[torch.Tensor, Dict[str, float]]: 
            A tuple of the loss and a dictionary of metrics.
        """
        total_loss = torch.tensor(0.0, device=outputs.device)
        coordinate_loss = torch.tensor(0.0, device=outputs.device)
        orientation_loss = torch.tensor(0.0, device=outputs.device)
        angle_loss = torch.tensor(0.0, device=outputs.device)
        coordinate_velocity_loss = torch.tensor(0.0, device=outputs.device)
        angular_velocity_loss = torch.tensor(0.0, device=outputs.device)

        for i in range(outputs.shape[1]):
            loss = (outputs[:, i] - targets[:, i]) ** 2

            # TODO: Write a more sophisticated loss function?
            if i in (0, 1, 2):  # coordinates of the torso (center)
                loss /= 5
                coordinate_loss += loss.mean()
            elif i in (3, 4, 5, 6):  # orientations of the torso (center)
                loss /= 0.2
                orientation_loss += loss.mean()
            elif i in (7, 8, 9, 10, 11, 12, 13, 14):  # angles between the torso and the links
                loss /= 0.5
                angle_loss += loss.mean()
            elif i in (15, 16, 17, 18, 19, 20):  # coordinate and coordinate angular velocities of the torso (center)
                loss /= 2
                coordinate_velocity_loss += loss.mean()
            elif i in (21, 22, 23, 24, 25, 26, 27, 28):  # angular velocities of the angles between the torso and the links
                loss /= 5
                angular_velocity_loss += loss.mean()

            total_loss += loss.mean()

        total_loss /= outputs.shape[1]
        coordinate_loss /= 3
        orientation_loss /= 4
        angle_loss /= 8
        coordinate_velocity_loss /= 6
        angular_velocity_loss /= 8
        metrics = {
            'coordinate_loss': coordinate_loss.item(), 
            'orientation_loss': orientation_loss.item(), 
            'angle_loss': angle_loss.item(), 
            'coordinate_velocity_loss': coordinate_velocity_loss.item(), 
            'angular_velocity_loss': angular_velocity_loss.item()
        }
       
        return total_loss, metrics
