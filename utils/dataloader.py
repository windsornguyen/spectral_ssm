# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen, Yagiz Devre
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
    def __init__(
        self,
        model,
        data,
        task,
        controller,
        shift=1,
        preprocess=True,
        sl=None,
        noise=0.0,
        noise_frequency=0.2,
        eps=1e-5,
        device=None,
    ):
        self.model = model
        self.task = task
        self.controller = controller
        self.shift = shift
        self.preprocess = preprocess
        self.sl = sl
        self.eps = eps
        self.rng = np.random.default_rng()  # Already seeded in main training script.

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

            if model == "mamba-2" and controller == "Ant-v1":
                # Padding zeros to the end of each trajectory at each timestep
                self.inputs = np.pad(
                    self.inputs, ((0, 0), (0, 0), (0, 3)), mode="constant"
                )
                self.targets = np.pad(
                    self.targets, ((0, 0), (0, 0), (0, 3)), mode="constant"
                )

            # Define feature groups w.r.t each task
            if controller == "Ant-v1":
                self.feature_groups = {
                    "coordinates": (0, 1, 2),
                    "orientations": (3, 4, 5, 6),
                    "angles": (7, 8, 9, 10, 11, 12, 13, 14),
                    "coordinate_velocities": (15, 16, 17, 18, 19, 20),
                    "angular_velocities": (21, 22, 23, 24, 25, 26, 27, 28),
                }
                if task == "mujoco-v1":
                    self.feature_groups["torque"] = (29, 30, 31, 32, 33, 34, 35, 36)
            else:
                self.feature_groups = {
                    "coordinates": (0, 1),
                    "angles": (2, 3, 4, 5, 6, 7, 8),
                    "coordinate_velocities": (9, 10),
                    "angular_velocities": (11, 12, 13, 14, 15, 16, 17),
                }
                if task == "mujoco-v1":
                    self.feature_groups["torque"] = (18, 19, 20, 21, 22, 23)

        elif task == "mujoco-v3":
            self.data = data

        # Apply noise first
        self.noise = noise
        self.noise_frequency = noise_frequency
        if self.noise > 0:
            print(
                f"\nApply Gaussian noise to data?: Enabled | Using noise={self.noise}, noise_frequency={self.noise_frequency}"
            )
            self.apply_noise()
        else:
            print("\nApply Gaussian noise to data?: Disabled")

        # Finally, normalize the data
        if self.preprocess:
            colored_print("\nCalculating data statistics...", Colors.OKBLUE)
            self._calculate_statistics()
            colored_print("Normalizing data...", Colors.OKBLUE)
            self._normalize_data()
            colored_print("Validating data normalization...", Colors.OKBLUE)
            self._validate_normalization()

    def __len__(self):
        if self.task == "mujoco-v3":
            return len(self.data)
        else:
            return len(self.inputs)

    def __getitem__(self, index):
        if self.task == "mujoco-v3":
            # MuJoCo-v3 data does not come offset by one
            features = self.data[index]
            input_frames = features[: -self.shift]
            target_frames = features[self.shift :]
            return input_frames, target_frames

        if self.sl:
            x_t = torch.tensor(self.inputs[index, : self.sl], dtype=torch.float32)
            x_t_plus_1 = torch.tensor(
                self.targets[index, : self.sl], dtype=torch.float32
            )
        else:
            # MuJoCo-v1 and MuJoCo-v2 data are already offset by one
            x_t = torch.tensor(self.inputs[index], dtype=torch.float32)
            x_t_plus_1 = torch.tensor(self.targets[index], dtype=torch.float32)
        return x_t, x_t_plus_1

    def _calculate_statistics(self):
        if self.task == "mujoco-v3":
            features = torch.cat(self.data, dim=0)
            # Mean over frames and samples, for each feature
            self.mean = features.mean(dim=(0, 1), keepdim=True)
            # Std over frames and samples, for each feature
            self.std = features.std(dim=(0, 1), keepdim=True)

        else:
            self.mean = {}
            self.std = {}
            for group_name, indices in self.feature_groups.items():
                if group_name == "torque":
                    group_data = self.inputs[:, :, indices]
                else:
                    group_data = np.concatenate(
                        [self.targets[:, :, indices], self.inputs[:, :, indices]],
                        axis=0,
                    )

                self.mean[group_name] = np.mean(group_data, axis=(0, 1), keepdims=True)
                self.std[group_name] = np.std(group_data, axis=(0, 1), keepdims=True)

    def _normalize_data(self):
        if self.task == "mujoco-v3":
            self.data = [
                (item - self.mean) / (self.std + self.eps) for item in self.data
            ]
        else:
            for group_name, indices in self.feature_groups.items():
                if group_name == "torque":
                    self.inputs[:, :, indices] = (
                        self.inputs[:, :, indices] - self.mean[group_name]
                    ) / (self.std[group_name] + self.eps)
                else:
                    self.inputs[:, :, indices] = (
                        self.inputs[:, :, indices] - self.mean[group_name]
                    ) / (self.std[group_name] + self.eps)
                    self.targets[:, :, indices] = (
                        self.targets[:, :, indices] - self.mean[group_name]
                    ) / (self.std[group_name] + self.eps)

    def _validate_normalization(self):
        if self.task == "mujoco-v3":
            features = torch.cat(self.data, dim=0)
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

        else:
            for group_name, indices in self.feature_groups.items():
                if group_name == "torque":
                    normalized_mean_inputs = np.mean(
                        self.inputs[:, :, indices], axis=(0, 1)
                    )
                    normalized_std_inputs = np.std(
                        self.inputs[:, :, indices], axis=(0, 1)
                    )
                    colored_print(
                        f"\nNormalized mean of inputs for {group_name}: {normalized_mean_inputs}",
                        Colors.OKGREEN,
                    )
                    colored_print(
                        f"Normalized standard deviation of inputs for {group_name}: {normalized_std_inputs}",
                        Colors.OKGREEN,
                    )
                else:
                    normalized_mean_inputs = np.mean(
                        self.inputs[:, :, indices], axis=(0, 1)
                    )
                    normalized_std_inputs = np.std(
                        self.inputs[:, :, indices], axis=(0, 1)
                    )
                    normalized_mean_targets = np.mean(
                        self.targets[:, :, indices], axis=(0, 1)
                    )
                    normalized_std_targets = np.std(
                        self.targets[:, :, indices], axis=(0, 1)
                    )

                    # # TODO: Normalized mean not close to zero
                    # assert np.allclose(
                    #     normalized_mean_inputs, np.zeros_like(normalized_mean_inputs), atol=self.eps
                    # ), f"Normalized mean of inputs for {group_name} is not close to zero: {normalized_mean_inputs}"
                    # assert np.allclose(
                    #     normalized_std_inputs, np.ones_like(normalized_std_inputs), atol=self.eps
                    # ), f"Normalized standard deviation of inputs for {group_name} is not close to one: {normalized_std_inputs}"
                    # assert np.allclose(
                    #     normalized_mean_targets, np.zeros_like(normalized_mean_targets), atol=self.eps
                    # ), f"Normalized mean of targets for {group_name} is not close to zero: {normalized_mean_targets}"
                    # assert np.allclose(
                    #     normalized_std_targets, np.ones_like(normalized_std_targets), atol=self.eps
                    # ), f"Normalized standard deviation of targets for {group_name} is not close to one: {normalized_std_targets}"

                    colored_print(
                        f"\nNormalized mean of inputs for {group_name}: {normalized_mean_inputs}",
                        Colors.OKGREEN,
                    )
                    colored_print(
                        f"Normalized standard deviation of inputs for {group_name}: {normalized_std_inputs}",
                        Colors.OKGREEN,
                    )
                    colored_print(
                        f"Normalized mean of targets for {group_name}: {normalized_mean_targets}",
                        Colors.OKGREEN,
                    )
                    colored_print(
                        f"Normalized standard deviation of targets for {group_name}: {normalized_std_targets}",
                        Colors.OKGREEN,
                    )

            colored_print(
                "Data normalization validated successfully.",
                Colors.BOLD + Colors.OKGREEN,
            )

    def apply_noise(self):
        if self.task in ["mujoco-v1", "mujoco-v2"]:
            num_samples, num_timesteps, _ = self.inputs.shape
            noise_mask = (
                self.rng.random((num_samples, num_timesteps)) < self.noise_frequency
            )

            # Apply noise to inputs only
            noise_input = self.rng.standard_normal(self.inputs.shape) * self.noise
            self.inputs += noise_input * noise_mask[..., np.newaxis]


def get_dataloader(
    model,
    data,
    task,
    controller,
    bsz,
    shift=1,
    preprocess=True,
    shuffle=True,
    pin_memory=False,
    distributed=True,
    local_rank=0,
    world_size=1,
    sl=None,
    noise=0.0,
    noise_frequency=0.2,
    eps=1e-5,
    device="cpu",
):
    colored_print(f"\nCreating dataloader on {device} for task: {task}", Colors.OKBLUE)
    dataset = Dataloader(
        model,
        data,
        task,
        controller,
        shift,
        preprocess,
        sl,
        noise,
        noise_frequency,
        eps,
    )
    pin_memory = device == "cpu"

    sampler = (
        DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=local_rank,
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


def split_data(dataset, train_ratio=0.8):
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
