import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from models.stu.model import SSSM, SSSMConfigs
from torch.nn import MSELoss
from safetensors.torch import load_file


def main():
    torch.set_float32_matmul_precision("high")

    # Process command line flags
    parser = argparse.ArgumentParser(
        description="Inference script for sequence prediction"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="Ant-v1",
        choices=["Ant-v1", "HalfCheetah-v1", "Walker2D-v1"],
        help="Controller to use for the MuJoCo environment.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mujoco-v3",
        choices=["mujoco-v1", "mujoco-v2", "mujoco-v3"],
        help="Task to run inference on.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = "ssm_6l.pt"

    configs = SSSMConfigs(
        n_layers=2,
        n_embd=512,
        d_out=512,
        sl=300,
        scale=4,
        bias=False,
        dropout=0.10,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
        loss_fn=MSELoss(),
        controls={"task": args.task, "controller": args.controller},
    )

    # Load the test data
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        test_inputs = np.load(f"{base_path}/test_inputs.npy")
        test_targets = np.load(f"{base_path}/test_targets.npy")
        test_inputs = torch.from_numpy(test_inputs).float().to(device)
        test_targets = torch.from_numpy(test_targets).float().to(device)
    elif args.task == "mujoco-v3":
        test_data = torch.load(
            f"data/{args.task}/{args.controller}/{args.controller}_ResNet-18_test.pt",
            map_location=device,
        )
        test_inputs = test_data
        test_targets = test_data
    else:
        raise ValueError("Invalid task")

    random_inputs = torch.randn_like(test_inputs)
    random_targets = torch.randn_like(test_targets)

    # Initialize and load the model
    model = SSSM(configs).to(device)
    model = torch.compile(model)
    state_dict = load_file(model_path, device="cuda:0")
    model.load_state_dict(state_dict)
    model.eval()

    # Simple loop to print model predictions
    num_iterations = 5  # You can adjust this number
    sequence_length = 300  # Assuming this is the sequence length based on the configs

    with torch.no_grad():
        for i in range(num_iterations):
            # Get a batch of inputs and targets
            # inputs = test_inputs[i : i + 1, :sequence_length, :]
            # targets = test_targets[i : i + 1, :sequence_length, :]
            inputs = random_inputs[i : i + 1, :sequence_length, :]
            targets = random_targets[i : i + 1, :sequence_length, :]

            # Forward pass
            preds, (loss,) = model.forward(inputs, targets)

            print(f"\nPredictions for iteration {i+1}:")
            print(
                preds[0, -1, :5].cpu().numpy()
            )  # Print the last time step, first 5 features
            print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    main()
