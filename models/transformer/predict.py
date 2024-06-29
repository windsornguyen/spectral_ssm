# =============================================================================#
# Authors: Isabel Liu
# File: predict.py
# =============================================================================#

"""Prediction loop for Transformer sequence prediction."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d
from safetensors.torch import load_file

from models.transformer.model import Transformer, TransformerConfigs
from torch.nn import MSELoss

from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss


def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)


def plot_losses(losses, title, x_values=None, ylabel="Loss", color=None):
    if x_values is None:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses), label=title, color=color)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()


def main():
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
    # model_path = f"best_{args.controller}.safetensors"
    model_path = "transformer_Ant-v1.pt"
    
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        if args.controller == "Ant-v1":
            loss_fn = AntLoss()
        elif args.controller == "HalfCheetah-v1":
            loss_fn = HalfCheetahLoss()
        elif args.controller == "Walker2D-v1":
            loss_fn = Walker2DLoss()
        else:
            loss_fn = None
    else:
        loss_fn = MSELoss()

    configs = TransformerConfigs(
        n_layers=4,
        n_embd=29,
        n_head=1,
        sl=1000,
        scale=16,
        bias=False,
        dropout=0.10,
        use_dilated_attn=False,
        loss_fn=loss_fn,
        controls={"task": args.task, "controller": args.controller},
    )

    # Initialize and load the model
    model = Transformer(configs).to(device)
    # model = torch.compile(model)
    state_dict = load_file(model_path, device="cuda:0")
    # state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load the test data
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        test_inputs = np.load(f"{base_path}/val_inputs.npy")
        test_targets = np.load(f"{base_path}/val_targets.npy")
        test_inputs = torch.from_numpy(test_inputs).float().to(device)
        test_targets = torch.from_numpy(test_targets).float().to(device)
    elif args.task == "mujoco-v3":
        test_data = torch.load(
            f"data/{args.task}/{args.controller}/{args.controller}_ResNet-18_test.pt",
            map_location=device,
        )
        test_inputs = test_data
        test_targets = test_data  # For mujoco-v3, inputs and targets are the same
    else:
        raise ValueError("Invalid task")

    # Run inference
    num_preds = 5
    predicted_states = []
    losses = []
    metrics = {}
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        steps = 100     # number of steps to predict
        metrics = {
            key: [torch.zeros(steps, device=device) for _ in range(num_preds)]
            for key in [
                "coordinate_loss",
                'orientation_loss',
                "angle_loss",
                "coordinate_velocity_loss",
                "angular_velocity_loss",
            ]
        }

    with torch.no_grad():
        for i in range(num_preds):
            inputs = test_inputs[i : i + 1]
            targets = test_targets[i : i + 1]

            if args.task in ["mujoco-v1", "mujoco-v2"]:
                pred_states, (avg_loss, avg_metric, loss, metric) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=0,
                    steps=100,
                    ar_steps=1000,
                )

                for key in metrics:
                    metrics[key].append(metric[key])

            elif args.task == "mujoco-v3":
                pred_states, (avg_loss, loss) = model.predict_frames(
                    inputs=inputs,
                    targets=targets,
                    init=140,
                    steps=5,
                    ar_steps=300,
                )

            predicted_states.append(pred_states)
            losses.append(loss)


    predicted_states = torch.cat(predicted_states, dim=0)
    losses = torch.cat(losses, dim=0)
    print(f"metrics before concatenation: {metrics}")
    metrics = {key: torch.cat(value, dim=0) for key, value in metrics.items()}
    print(f"metrics after concatenation: {metrics}")
    # metrics = {key: torch.cat(value, dim=0) for key, value in metrics.items()}

    print(f"Shape of predicted states: {predicted_states.shape}")
    print(f"Shape of losses: {losses.shape}")
    for key, value in metrics.items():
        for i, tensor in enumerate(value):
            print(f"The shape of {key} for prediction {i} is: {tensor.shape}") # empty if mujoco-v1/v2

    # Print out predictions and check if they're all the same
    for i in range(num_preds):
        print(f"\n Prediction {i+1}:")
        print(predicted_states[i, :5, 0])  # Print first 5 time steps of first feature

    # Check if all predictions are the same
    all_same = True
    for i in range(1, num_preds):
        if not torch.allclose(predicted_states[0], predicted_states[i], atol=1e-6):
            all_same = False
            break

    print(f"\nAll predictions are the same: {all_same}")

    if all_same:
        print(
            "All predictions are identical. This might indicate an issue with the model or data processing."
        )
    else:
        print(
            "Predictions differ, which is expected for different inputs."
        )

     # TODO: Save predictions and ground truths
    print("saved prediction shape", predicted_states.shape)
    print(
        "saved ground truth shape",
        test_targets[:num_preds, : predicted_states.shape[1], :].shape,
    )
    np.save("transformer_predictions_Ant-v1", predicted_states.cpu().numpy())
    np.save(
        "transformer_ground_truths_Ant-v1",
        test_targets[:num_preds, : predicted_states.shape[1], :].cpu().numpy(),
    )
    print(
        "Predictions and ground truths saved to 'predictions.npy' and 'ground_truths.npy' respectively."
    )

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    num_rows = num_preds
    num_cols = 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 4 * num_rows))

    colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_preds)]

    for pred_idx in range(num_preds):
        time_steps = predicted_states.shape[1]
        print(f"Plotting prediction {pred_idx + 1} over {time_steps} time steps")

        # Plot the predicted states and ground truth states
        for feature_idx in range(3):  # Plot first three features
            axs[pred_idx, 0].plot(
                range(time_steps),
                test_targets[pred_idx, :time_steps, feature_idx].cpu().numpy(),
                label=f"Ground Truth {pred_idx+1}, Feature {feature_idx+1}",
                color=colors[pred_idx],
                linewidth=2,
                linestyle="--",
            )
            axs[pred_idx, 0].plot(
                range(time_steps),
                predicted_states[pred_idx, :, feature_idx].cpu().numpy(),
                label=f"Predicted {pred_idx+1}, Feature {feature_idx+1}",
                color=colors[pred_idx],
                linewidth=2,
            )

        axs[pred_idx, 0].set_title(f"Prediction {pred_idx+1}: Predicted vs Ground Truth")
        axs[pred_idx, 0].set_xlabel("Time Step")
        axs[pred_idx, 0].set_ylabel("State")
        axs[pred_idx, 0].legend()
        axs[pred_idx, 0].grid(True)

        # Plot the losses (scaled up by 100)
        scaled_losses = smooth_curve(losses[pred_idx].cpu().numpy())
        axs[pred_idx, 1].plot(
            range(time_steps),
            scaled_losses,
            color=colors[pred_idx],
            linewidth=2,
        )
        axs[pred_idx, 1].set_title(f"Prediction {pred_idx+1}: Loss")
        axs[pred_idx, 1].set_xlabel("Time Step")
        axs[pred_idx, 1].set_ylabel("Loss (scaled)")
        axs[pred_idx, 1].grid(True)

        if args.task in ["mujoco-v1", "mujoco-v2"]:
            # Plot the metrics
            for key in metrics:
                metric_values = metrics[key][pred_idx].cpu().numpy()
                axs[pred_idx, 1].plot(
                    range(time_steps),
                    smooth_curve(metric_values),
                    label=f"{key}",
                    linewidth=2,
                )
            axs[pred_idx, 1].legend()

    plt.tight_layout()
    plt.savefig(
        f"results/{args.controller}_{args.task}_predictions_tr.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
