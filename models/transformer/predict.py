# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: predict.py
# =============================================================================#

"""Prediction loop for Transformer sequence prediction."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from safetensors.torch import load_file
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
    model_path = f"transformer_{args.controller}_{args.task}.pt"

    if args.task == "mujoco-v1":
        sl = 1000
        if args.controller == "Ant-v1":
            n_embd, d_out = 37, 29
            n_heads = 1
            loss_fn = AntLoss()
        elif args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
            n_embd, d_out = 24, 18
            n_heads = 8
            loss_fn = (
                HalfCheetahLoss()
                if args.controller == "HalfCheetah-v1"
                else Walker2DLoss()
            )
        else:
            n_embd, d_out, loss_fn = None, None, None
    elif args.task == "mujoco-v2":
        sl = 1000
        n_heads = 1
        if args.controller == "Ant-v1":
            n_embd, d_out = 29, 29
            loss_fn = AntLoss()
        elif args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
            n_embd, d_out = 18, 18
            loss_fn = (
                HalfCheetahLoss()
                if args.controller == "HalfCheetah-v1"
                else Walker2DLoss()
            )
        else:
            n_embd, d_out, loss_fn = None, None, None
    elif args.task == "mujoco-v3":
        n_heads = 8
        sl, n_embd, d_out = 300, 512, 512
        loss_fn = MSELoss()
    else:
        raise ValueError("Invalid task")

    configs = TransformerConfigs(
        n_layers=2,
        n_embd=n_embd,
        n_heads=n_heads,
        sl=sl,
        scale=16,
        bias=False,
        dropout=0.0,
        dilated_attn=False,
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
    init = 950 if args.task in ["mujoco-v1", "mujoco-v2"] else 295
    steps = len(test_inputs[1]) - init

    with torch.no_grad():
        for i in range(num_preds):
            inputs = test_inputs[i : i + 1]
            targets = test_targets[i : i + 1]

            if args.task in ["mujoco-v1", "mujoco-v2"]:
                (
                    pred_states,
                    (avg_loss, trajectory_losses),
                ) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,  # Use the first 100 steps as context
                    steps=steps,  # Predict the next steps
                    rollout_steps=20,
                )

            elif args.task == "mujoco-v3":
                (
                    pred_states,
                    (avg_loss, trajectory_losses),
                ) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,
                    steps=steps,  # Predict the next 5 steps
                    rollout_steps=1,
                )

            predicted_states.append(pred_states)
            losses.append(trajectory_losses)

    predicted_states = torch.cat(predicted_states, dim=0)
    losses = torch.cat(losses, dim=0)

    print(f"Shape of predicted states: {predicted_states.shape}")
    print(f"Shape of losses: {losses.shape}")

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
        print("Predictions differ, which is expected for different inputs.")

    # Save predictions and ground truths
    print("Saved prediction shape:", predicted_states.shape)
    print(
        "Saved ground truth shape:",
        test_targets[:num_preds, -predicted_states.shape[1] :, :].shape,
    )
    print("Saved losses shape:", losses.shape)
    np.save(
        f"transformer_{args.controller}_{args.task}_predictions.npy",
        predicted_states.cpu().numpy(),
    )
    np.save(
        f"transformer_{args.controller}_{args.task}_ground_truths.npy",
        test_targets[:num_preds, -predicted_states.shape[1] :, :].cpu().numpy(),
    )
    np.save(f"transformer_{args.controller}_{args.task}_losses.npy", losses.cpu().numpy())
    print(
        f"Predictions, ground truths, and losses saved to 'transformer_{args.controller}_{args.task}_predictions.npy', 'transformer_{args.controller}_{args.task}_ground_truths.npy', and 'transformer_{args.controller}_{args.task}_losses.npy' respectively."
    )

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)
    colors = plt.cm.viridis(np.linspace(0, 1, num_preds))

    fig = plt.figure(figsize=(20, 8 * num_preds))
    gs = GridSpec(
        num_preds, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.3, hspace=0.4
    )

    for pred_idx in range(num_preds):
        print(f"Plotting prediction {pred_idx + 1}")

        # Plot predicted states vs ground truth
        ax1 = fig.add_subplot(gs[pred_idx, 0])
        feature_idx = 0

        # Plot ground truth
        ax1.plot(
            range(init, init + steps - 1),
            test_targets[pred_idx, init : init + steps - 1, feature_idx].cpu().numpy(),
            label="Ground Truth",
            color="black",
            linewidth=2,
            linestyle="--",
        )

        # Plot prediction
        ax1.plot(
            range(init, init + steps - 1),
            predicted_states[pred_idx, : steps - 1, feature_idx].cpu().numpy(),
            label="Predicted",
            color=colors[pred_idx],
            linewidth=2,
        )

        ax1.set_title(f"Prediction {pred_idx+1}: Predicted vs Ground Truth")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("State Value")
        ax1.legend()

        # Plot losses and metrics
        ax2 = fig.add_subplot(gs[pred_idx, 1])

        # Plot losses
        ax2.plot(
            range(steps - 1),
            smooth_curve(losses[pred_idx, : steps - 1].cpu().numpy()),
            label="Total Loss",
            color="black",
            linewidth=2,
        )

        ax2.set_title(f"Prediction {pred_idx+1}: Loss and Metrics")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Value")
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_yscale("log")  # Use log scale for better visibility

    plt.suptitle(
        f"Transformer Predictions for {args.controller} on {args.task}\n",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # TODO: Add existok / make if non-existent (results/) directory
    plt.savefig(
        f"results/transformer_{args.controller}_{args.task}_predictions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
