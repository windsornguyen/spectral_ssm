import numpy as np
import torch
from data.physics.physics_data import get_dataloader as get_physics_dataloader
from utils.dataloader import get_dataloader as get_custom_dataloader


def compare_dataloaders(physics_dataloader, custom_dataloader, num_batches=5):
    print("Comparing dataloaders...")

    for batch_idx, (physics_batch, custom_batch) in enumerate(
        zip(physics_dataloader, custom_dataloader, strict=True)
    ):
        if batch_idx >= num_batches:
            break

        physics_input, physics_target = physics_batch
        custom_input, custom_target = custom_batch

        print(f"\nBatch {batch_idx + 1}:")
        print(
            f"Physics input shape: {physics_input.shape}, Custom input shape: {custom_input.shape}"
        )
        print(
            f"Physics target shape: {physics_target.shape}, Custom target shape: {custom_target.shape}"
        )

        input_match = torch.allclose(physics_input, custom_input, atol=1e-6)
        target_match = torch.allclose(physics_target, custom_target, atol=1e-6)

        print(f"Inputs match: {input_match}")
        print(f"Targets match: {target_match}")

        if not input_match or not target_match:
            print("Mismatch detected. Printing first few elements:")
            print("Physics input:", physics_input[0, :5])
            print("Custom input:", custom_input[0, :5])
            print("Physics target:", physics_target[0, :5])
            print("Custom target:", custom_target[0, :5])

            input_diff = (physics_input - custom_input).abs().max().item()
            target_diff = (physics_target - custom_target).abs().max().item()
            print(f"Max input difference: {input_diff}")
            print(f"Max target difference: {target_diff}")


def main():
    # Assume these are the paths to your data files
    input_file = "/scratch/gpfs/mn4560/ssm/data/mujoco-v2/Ant-v1/train_inputs.npy"
    target_file = "/scratch/gpfs/mn4560/ssm/data/mujoco-v2/Ant-v1/train_targets.npy"

    # Load data for custom dataloader
    inputs = np.load(input_file)
    targets = np.load(target_file)
    data = {"inputs": input_file, "targets": target_file}

    # Create dataloaders
    batch_size = 32
    physics_dataloader = get_physics_dataloader(
        input_file, target_file, batch_size, device="cpu", distributed=False
    )
    custom_dataloader = get_custom_dataloader(
        data,
        task="mujoco-v2",
        bsz=batch_size,
        distributed=False,
        preprocess=False,
        device="cpu",
    )

    # Compare dataloaders
    compare_dataloaders(physics_dataloader, custom_dataloader)


if __name__ == "__main__":
    main()
