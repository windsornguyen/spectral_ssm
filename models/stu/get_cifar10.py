# =============================================================================#
# Authors: Windsor Nguyen, Dwaipayan Saha
# File: cifar10.py
# =============================================================================#

"""Data pipeline for CIFAR-10."""

import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class Transform:
    """Class for normalizing images with dataset-specific statistics."""

    def __init__(self):
        self.means = torch.tensor([0.49139968, 0.48215841, 0.44653091])
        self.stds = torch.tensor([0.24703223, 0.24348513, 0.26158784])

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize the input image.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        return (img - self.means[:, None, None]) / self.stds[:, None, None]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Prepare the transformation pipeline for the CIFAR-10 dataset.

    Args:
        is_train (bool, optional): Flag to indicate whether to apply
            training-specific transforms.

    Returns:
        transforms.Compose: A composition of transformations.
    """
    basic = [transforms.ToTensor(), Transform()]
    if train:
        augmentations = [
            # Mild cropping and padding
            transforms.RandomCrop(32, padding=4),
            # Mirror image across y-axis
            transforms.RandomHorizontalFlip(),
            # Mild color adjustments
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1
            ),
            # Small rotations
            transforms.RandomRotation(degrees=5),
            # Random cutout
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0
            ),
        ]
        return transforms.Compose(basic + augmentations)
    else:
        return transforms.Compose(basic)


def get_dataset(split: str) -> Dataset:
    """Retrieve the CIFAR-10 dataset with preprocessing and augmentation.

    Args:
        split (str): The dataset split to use, either 'train' or 'test'.

    Returns:
        Dataset: The CIFAR-10 dataset with the specified split and
            appropriate preprocessing.
    """
    train = split == 'train'
    transform = get_transforms(train)
    dataset = datasets.CIFAR10(
        root='./data', train=train, download=True, transform=transform
    )

    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    distributed: bool = True,
    rank: int = 0,
    num_replicas: int = 1,
    num_workers: int = 1,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to create a DataLoader for.
        batch_size (int): The number of samples per batch to load.
        distributed (bool, optional): If True, use DistributedSampler for
            distributed training.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        num_workers (int, optional): Number of subprocesses to use for data
            loading.
        pin_memory (bool, optional): If True, the data loader will copy
            Tensors into CUDA pinned memory before returning them.

    Returns:
        DataLoader: A DataLoader instance for the given dataset.
    """
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=num_replicas, 
        rank=rank
    ) if distributed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=(sampler is None),
    )
