# =============================================================================#
# Authors: Windsor Nguyen, Yagiz Devre, Isabel Liu
# File: dataloader.py
# =============================================================================#

"""Custom dataloader for loading sequence data in a distributed manner."""

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.colors import Colors, colored_print


# TODO: Write generic dataset downloading and saving script for the user.
class Dataloader(Dataset):
    def __init__(self, data, task, shift=1, preprocess=True, eps=1e-7):
        self.task = task
        self.shift = shift
        self.preprocess = preprocess
        self.eps = eps

        if task in ["mujoco-v1", "mujoco-v2"]:
            # For .txt files
            if isinstance(data["inputs"], str) and isinstance(data["targets"], str):
                self.inputs = np.load(data["inputs"])
                self.targets = np.load(data["targets"])
            # For .npy files
            elif isinstance(data["inputs"], torch.Tensor) and isinstance(
                data["targets"], torch.Tensor
            ):
                self.inputs = data["inputs"].cpu().numpy()
                self.targets = data["targets"].cpu().numpy()
            else:
                raise ValueError("Invalid data format for mujoco-v1 or mujoco-v2 tasks")
            self.data = None
        else:
            self.data = data
            self.inputs = None
            self.targets = None

        if self.preprocess and self.task == "mujoco-v3":
            colored_print("Calculating data statistics...", Colors.OKBLUE)
            self._calculate_statistics()
            colored_print("Normalizing data...", Colors.OKBLUE)
            self._normalize_data()
            colored_print("Validating data normalization...", Colors.OKBLUE)
            self._validate_normalization()

    def __len__(self):
        return len(self.inputs) if self.inputs is not None else len(self.data)

    def __getitem__(self, index):
        if self.task in ["mujoco-v1", "mujoco-v2"]:
            x_t = torch.tensor(self.inputs[index], dtype=torch.float32)
            x_t_plus_1 = torch.tensor(self.targets[index], dtype=torch.float32)
            return x_t, x_t_plus_1
        elif self.task == "mujoco-v3":
            features = self.data[index]
        else:
            features = torch.cat(
                (self.data[index]["x_t"], self.data[index]["x_t_plus_1"]),
                dim=-1,
            )

        input_frames = features[: -self.shift]
        target_frames = features[self.shift :]

        return input_frames, target_frames

    def _calculate_statistics(self):
        if self.task in ["mujoco-v1", "mujoco-v2"]:
            features = np.concatenate((self.inputs, self.targets), axis=-1)
        elif self.task == "mujoco-v3":
            features = torch.cat(self.data, dim=0)

        # Mean over frames and samples, for each feature
        self.mean = features.mean(dim=(0, 1), keepdim=True)

        # Std over frames and samples, for each feature
        self.std = features.std(dim=(0, 1), keepdim=True)

    def _normalize_data(self):
        self.inputs = (self.inputs - self.mean.numpy()) / (self.std.numpy() + self.eps)
        self.targets = (self.targets - self.mean.numpy()) / (
            self.std.numpy() + self.eps
        )

    def _validate_normalization(self):
        if self.task in ["mujoco-v1", "mujoco-v2"]:
            features = np.concatenate((self.inputs, self.targets), axis=-1)
            features = torch.tensor(features)
        elif self.task == "mujoco-v3":
            features = torch.cat(self.data, dim=0)
        else:
            features = torch.cat(
                [
                    torch.cat(
                        (self.data[i]["x_t"], self.data[i]["x_t_plus_1"]),
                        dim=-1,
                    )
                    for i in range(len(self.data))
                ],
                dim=0,
            )

        normalized_mean = features.mean(dim=(0, 1))
        normalized_std = features.std(dim=(0, 1))

        assert torch.allclose(
            normalized_mean, torch.zeros_like(normalized_mean), atol=self.eps
        ), f"Normalized mean is not close to zero: {normalized_mean}"
        assert torch.allclose(
            normalized_std, torch.ones_like(normalized_std), atol=self.eps
        ), f"Normalized standard deviation is not close to one: {normalized_std}"

        colored_print(f"\nNormalized mean: {normalized_mean}", Colors.OKGREEN)
        colored_print(
            f"Normalized standard deviation: {normalized_std}", Colors.OKGREEN
        )
        colored_print(
            "Data normalization validated successfully.",
            Colors.BOLD + Colors.OKGREEN,
        )


def get_dataloader(
    data,
    task,
    bsz,
    shift=1,
    preprocess=True,
    shuffle=True,
    pin_memory=False,
    distributed=True,
    rank=0,
    world_size=1,
    device="cpu",
):
    colored_print(f"\nCreating dataloader on {device} for task: {task}", Colors.OKBLUE)
    dataset = Dataloader(data, task, shift, preprocess)
    pin_memory = device == "cpu"

    sampler = (
        DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        if distributed
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=(shuffle and sampler is None),
        pin_memory=pin_memory,
        sampler=sampler,
    )

    colored_print("Dataloader created successfully.", Colors.OKGREEN)
    return dataloader


def split_data(dataset, train_ratio=0.8, random_seed=1337):
    # TODO: Not sure if we need this, experiment to see if data is the same without
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)

    if isinstance(dataset, Dataloader) and dataset.task in ["mujoco-v1", "mujoco-v2"]:
        num_samples = len(dataset)
        indices = torch.randperm(num_samples)
        train_size = int(train_ratio * num_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_data = {
            "inputs": dataset.inputs[train_indices],
            "targets": dataset.targets[train_indices],
        }
        val_data = {
            "inputs": dataset.inputs[val_indices],
            "targets": dataset.targets[val_indices],
        }
    else:
        indices = torch.randperm(len(dataset))
        shuffled_data = [dataset[i] for i in indices]

        num_train = int(len(dataset) * train_ratio)
        train_data = shuffled_data[:num_train]
        val_data = shuffled_data[num_train:]

    colored_print(
        f"\nData split into {len(train_data)} training samples and {len(val_data)} validation samples.",
        Colors.OKBLUE,
    )

    return train_data, val_data
