# ==============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: optimizer.py
# ==============================================================================#


"""AdamW with linear warmup and cosine decay."""

import torch
from torch.optim import AdamW


# TODO: Deprecated in favor of the get_optimizer and get_lr in experiment.py. Remove this file.
class WarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        start_val: float,
        min_lr: float,
        lr: float,
        num_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """Initialize a cosine decay schedule with warmup.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to schedule.
            start_val (float): The value to start at.
            min_lr (float): The minimum value to decay to.
            lr (float): The peak value to reach.
            num_steps (int): The total number of steps to decay over.
            warmup_steps (int): The number of steps to warmup for.
            last_epoch (int): The index of the last epoch. Default is -1.
        """
        self.start_val = start_val
        self.min_lr = min_lr
        self.lr = lr
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get learning rate for a given step.

        Returns:
            list[float]: The learning rate for each parameter group.
        """
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [
                self.start_val + warmup_factor * (self.lr - self.start_val)
                for _ in self.base_lrs
            ]

        # Cosine annealing
        cos_factor = 0.5 * (
            1
            + torch.cos(
                torch.tensor(
                    torch.pi
                    * (self.last_epoch - self.warmup_steps)
                    / (self.num_steps - self.warmup_steps)
                )
            )
        )
        return [
            self.min_lr + (self.lr - self.min_lr) * cos_factor for _ in self.base_lrs
        ]

    def get_last_lr(self) -> list[float]:
        """Get last computed learning rate by the scheduler.

        Returns:
            list[float]: The last computed learning rate for each parameter group.
        """
        return self._last_lr


def get_optimizer(
    model: torch.nn.Module,
    num_steps: int,
    warmup_steps: int,
    learning_rate: float,
    weight_decay: float,
    **kwargs,
) -> tuple[torch.optim.AdamW, WarmupCosineDecay]:
    """Get the AdamW optimizer with warmup cosine decay scheduler.

    Args:
        model (torch.nn.Module): The model to optimize.
        num_steps (int): The total number of steps to decay over.
        warmup_steps (int): The number of steps to warmup for.
        learning_rate (float): The peak learning rate to reach.
        weight_decay (float): The weight decay for default parameters.
        **kwargs: Additional keyword arguments specific to each model.

    Returns:
        tuple[torch.optim.AdamW, WarmupCosineDecay]: 
            The AdamW optimizer and the warmup cosine decay scheduler.
    """
    param_groups = [
        {
            "params": model.parameters(), 
            "lr": learning_rate, 
            "weight_decay": weight_decay
        }
    ]

    if "stu" in model.__class__.__name__.lower():
        m_y_params = []
        default_params = []
        for name, param in model.named_parameters():
            if name.startswith("m_y"):
                m_y_params.append(param)
            else:
                default_params.append(param)

        param_groups = [
            {"params": default_params, "lr": learning_rate, "weight_decay": weight_decay},
            {
                "params": m_y_params,
                "lr": kwargs.get("m_y_learning_rate", 5e-5),
                "weight_decay": kwargs.get("m_y_weight_decay", 0),
            },
        ]

    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=True, # Can provide faster convergence in some cases.
        fused=False, # Combine weight decay and update steps into one operation.
        # TODO: Fix this bc fused needs CUDA so do it Andrej's way
    )

    scheduler = WarmupCosineDecay(
        optimizer,
        start_val=1e-7,
        min_lr=1e-7,
        lr=learning_rate,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
    )

    return optimizer, scheduler
