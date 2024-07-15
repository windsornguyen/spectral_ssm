# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: predict.py
# =============================================================================#

"""Prediction loop for Hybrid sequence prediction."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d
from safetensors.torch import load_file
import seaborn as sns
from matplotlib.gridspec import GridSpec

from models.hybrid.model import SpectralHybrid, SpectralHybridConfigs

from torch.nn import HuberLoss, MSELoss
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss

from utils.dist_utils import get_data_parallel_group


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
    # dilated attention
    parser.add_argument(
        "--dilated_attn",
        type=bool,
        default=False,
        help="Whether to use dilated attention. Defaults to False.",
    )
    parser.add_argument(
        "--segment_lengths",
        type=int,
        nargs='+',
        default=[128, 256, 512],
        help="Segment lengths for dilated attention. Defaults to [128, 256, 512].",
    )
    parser.add_argument(
        "--dilated_ratios",
        type=int,
        nargs='+',
        default=[1, 2, 4],
        help="Dilation ratios for dilated attention. Defaults to [1, 2, 4].",
    )
    parser.add_argument(
        "--seq_parallel",
        type=bool,
        default=True,
        help="Whether to use sequence parallelism. Defaults to True.",
    )
    parser.add_argument(
        "--xpos_rel_pos",
        type=bool,
        default=True,
        help="Whether to use relative positional embeddings. Defaults to True.",
    )
    parser.add_argument(
        "--xpos_scale_base",
        type=int,
        default=512,
        help="Scale base for positional embeddings. Defaults to 512.",
    )
    parser.add_argument(
        "--rms_norm_eps",
        type=float,
        default=1e-5,
        help="Epsilon for root mean square normalization. Defaults to 1e-5.",
    )
    parser.add_argument(
        "--multiway",
        type=bool,
        default=False,
        help="Whether to use multiway attention. Defaults to False.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = f"hybrid_{args.controller}_{args.task}.pt"

    if args.task != "mujoco-v3":
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

    # Task-specific hyperparameters
    if args.task == "mujoco-v1":
        n_embd: int = 24 if args.controller != "Ant-v1" else 37
        n_heads: int = 8 if args.controller != "Ant-v1" else 1
        d_in = n_embd  # TODO: d_in is not exactly the same as n_embd
        d_out = d_in    # before projection d_in = d_out
        d_proj: int = 18 if args.controller != "Ant-v1" else 29
        sl: int = 1000

    elif args.task == "mujoco-v2":
        n_embd: int = 18 if args.controller != "Ant-v1" else 29
        n_heads: int = 9 if args.controller != "Ant-v1" else 1
        d_in = n_embd  # TODO: d_in is not exactly the same as n_embd
        d_out = n_embd
        d_proj = n_embd
        sl: int = 1000

    elif args.task == "mujoco-v3":
        RESNET_D_OUT: int = 512  # ResNet-18 output dim
        RESNET_FEATURE_SIZE: int = 1
        d_out: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        n_embd: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        n_heads: int = 16
        d_in = n_embd  # TODO: d_in is not exactly the same as n_embd
        d_proj = n_embd
        sl: int = 300

    configs = SpectralHybridConfigs(
        # STU settings
        d_in=d_in,
        d_out=d_out,
        d_proj=d_proj,
        num_eigh=16,
        k_y=2,
        k_u=3,
        learnable_m_y=True,
        alpha=0.9,
        use_ar_y=False,
        use_ar_u=True,
        use_hankel_L=False,

        # Transformer settings
        n_embd=n_embd,
        n_heads=n_heads,
        sub_rn=True,
        flash_attn=True,
        
        # MoE
        moe=True,
        num_experts=3,
        num_experts_per_timestep=2,

        # Dilated Attention
        dilated_attn=args.dilated_attn,
        segment_lengths=args.segment_lengths,
        dilated_ratios=args.dilated_ratios,
        seq_parallel=args.seq_parallel,
        xpos_rel_pos=args.xpos_rel_pos,
        xpos_scale_base=args.xpos_scale_base,
        rms_norm_eps=args.rms_norm_eps,
        multiway=args.multiway,
        
        # General training settings
        sl=sl,
        n_layers=2,
        scale=4,
        bias=False,
        dropout=0.0,
        loss_fn=loss_fn,
        controls={"task": args.task, "controller": args.controller},
        device=device,
    )

    # Initialize and load the model
    model = SpectralHybrid(configs).to(device)
    # model = torch.compile(model)
    state_dict = load_file(model_path, device="cuda:0")
    model.load_state_dict(state_dict)
    model.eval()

    # Load the test data
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        test_inputs = np.load(f"{base_path}/val_inputs.npy")
        test_targets = np.load(f"{base_path}/val_targets.npy")

        # Define feature groups w.r.t each task
        if args.controller == "Ant-v1":
            feature_groups = {
                "coordinates": (0, 1, 2),
                "orientations": (3, 4, 5, 6),
                "angles": (7, 8, 9, 10, 11, 12, 13, 14),
                "coordinate_velocities": (15, 16, 17, 18, 19, 20),
                "angular_velocities": (21, 22, 23, 24, 25, 26, 27, 28),
            }
        else:
            feature_groups = {
                "coordinates": (0, 1),
                "angles": (2, 3, 4, 5, 6, 7, 8),
                "coordinate_velocities": (9, 10),
                "angular_velocities": (11, 12, 13, 14, 15, 16, 17),
            }

        # Normalize the test data
        mean = {}
        std = {}
        for group_name, indices in feature_groups.items():
            group_data = np.concatenate([test_targets[:, :, indices], test_inputs[:, :, indices]], axis=0)
            mean[group_name] = np.mean(group_data, axis=(0, 1), keepdims=True)
            std[group_name] = np.std(group_data, axis=(0, 1), keepdims=True)

        for group_name, indices in feature_groups.items():
            test_inputs[:, :, indices] = (test_inputs[:, :, indices] - mean[group_name]) / (std[group_name] + 1e-6)
            test_targets[:, :, indices] = (test_targets[:, :, indices] - mean[group_name]) / (std[group_name] + 1e-6)
        
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
                    init=init,
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
                    init=init,  # Use the first 295 steps as context
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
    test_targets = torch.roll(test_targets, shifts=1, dims=1) # shift ground truth by 1
    print(
        "Saved ground truth shape:",
        test_targets[:num_preds, -predicted_states.shape[1] :, :].shape,
    )
    print("Saved losses shape:", losses.shape)
    np.save(
        f"hybrid_{args.controller}_{args.task}_predictions.npy",
        predicted_states.cpu().numpy(),
    )
    np.save(
        f"hybrid_{args.controller}_{args.task}_ground_truths.npy",
        test_targets[:num_preds, -predicted_states.shape[1] :, :].cpu().numpy(),
    )
    np.save(f"hybrid_{args.controller}_{args.task}_losses.npy", losses.cpu().numpy())
    print(
        f"Predictions, ground truths, and losses saved to 'hybrid_{args.controller}_{args.task}_predictions.npy', 'hybrid_{args.controller}_{args.task}_ground_truths.npy', and 'hybrid_{args.controller}_{args.task}_losses.npy' respectively."
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
            range(init, init + steps),
            test_targets[pred_idx, init : init + steps, feature_idx].cpu().numpy(),
            label="Ground Truth",
            color="black",
            linewidth=2,
            linestyle="--",
        )

        # Plot prediction
        ax1.plot(
            range(init, init + steps),
            predicted_states[pred_idx, : steps, feature_idx].cpu().numpy(),
            label="Predicted",
            color=colors[pred_idx],
            linewidth=2,
        )

        ax1.set_title(f"Prediction {pred_idx+1}: Predicted vs Ground Truth")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("State Value")
        ax1.legend()

        # Plot losses
        ax2 = fig.add_subplot(gs[pred_idx, 1])

        ax2.plot(
            range(steps),
            smooth_curve(losses[pred_idx, : steps].cpu().numpy()),
            label="Total Loss",
            color="black",
            linewidth=2,
        )

        ax2.set_title(f"Prediction {pred_idx+1}: Losses")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Value")
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_yscale("log")  # Use log scale for better visibility

    plt.suptitle(
        f"Spectral Hybrid Predictions for {args.controller} on {args.task}\n",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # TODO: Add existok / make if non-existent (results/) directory
    plt.savefig(
        f"results/hybrid_{args.controller}_{args.task}_predictions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
