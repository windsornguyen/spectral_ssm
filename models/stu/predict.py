# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: predict.py
# =============================================================================#

"""Prediction loop for STU sequence prediction."""

import argparse
import torch
from torch.nn import HuberLoss, MSELoss
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d
from safetensors.torch import load_file
import seaborn as sns
from matplotlib.gridspec import GridSpec

from models.stu.model import SpectralSSM, SpectralSSMConfigs
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from utils.dataloader import get_dataloader


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
    parser.add_argument(
        "--bsz",
        type=int,
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=1,
        help="Number of time steps to shift the targets from the inputs by.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = f"sssm_{args.controller}_{args.task}.pt"

    if args.task == "mujoco-v1":
        sl = 1000
        if args.controller == "Ant-v1":
            n_embd, d_out, d_proj = 37, 37, 29
            loss_fn = AntLoss()
        elif args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
            n_embd, d_out, d_proj = 24, 24, 18
            loss_fn = (
                HalfCheetahLoss()
                if args.controller == "HalfCheetah-v1"
                else Walker2DLoss()
            )
        else:
            n_embd, d_out, d_proj, loss_fn = None, None, None, None
    elif args.task == "mujoco-v2":
        sl = 1000
        if args.controller == "Ant-v1":
            n_embd, d_out, d_proj = 29, 29, 29
            loss_fn = AntLoss()
        elif args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
            n_embd, d_out, d_proj = 18, 18, 18
            loss_fn = (
                HalfCheetahLoss()
                if args.controller == "HalfCheetah-v1"
                else Walker2DLoss()
            )
        else:
            n_embd, d_out, d_proj, loss_fn = None, None, None, None
    elif args.task == "mujoco-v3":
        sl, n_embd, d_out, d_proj = 300, 512, 512, 512
        loss_fn = MSELoss()
    else:
        raise ValueError("Invalid task")

    configs = SpectralSSMConfigs(
        n_layers=2,
        n_embd=n_embd,
        d_in=n_embd,  # TODO: Fix later, d_in \neq n_embd
        d_out=d_out,
        d_proj=d_proj,
        sl=sl,
        scale=4,
        bias=False,
        dropout=0.0,
        num_eigh=16,
        k_y=2,
        k_u=3,
        learnable_m_y=True,
        alpha=0.9,
        use_ar_y=False,
        use_ar_u=True,
        use_hankel_L=False,
        moe=True,
        num_experts=3,
        num_experts_per_timestep=2,
        loss_fn=loss_fn,
        controls={"task": args.task, "controller": args.controller},
        device=device,
    )

    # Initialize and load the model
    model = SpectralSSM(configs).to(device)
    # model = torch.compile(model)
    state_dict = load_file(model_path, device="cuda:0")
    model.load_state_dict(state_dict)
    model.eval()

    # Load the test data
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        test_data = {
            "inputs": f"{base_path}/val_inputs.npy",
            "targets": f"{base_path}/val_targets.npy"
        }
    elif args.task == "mujoco-v3":
        test_data = torch.load(
            f"data/{args.task}/{args.controller}/{args.controller}_ResNet-18_val.pt",
            map_location=device,
        )
    else:
        raise ValueError("Invalid task")

    # Get test data loader
    test_loader = get_dataloader(
        model="spectral_ssm",
        data=test_data,
        task=args.task,
        controller=args.controller,
        bsz=args.bsz,
        shift=args.shift,
        preprocess=True,
        shuffle=False,
        pin_memory=True,
        distributed=False,
        local_rank=0,
        world_size=1,
        device=device,
    )

    # Run inference
    num_preds = 500
    predicted_states = []
    losses = []
    init = 950 if args.task in ["mujoco-v1", "mujoco-v2"] else 295
    
    # Get the first batch to determine the steps
    first_batch = next(iter(test_loader))
    steps = first_batch[0].shape[1] - init

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= num_preds:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            if args.task in ["mujoco-v1", "mujoco-v2"]:
                (
                    pred_states,
                    (avg_loss, trajectory_losses),
                ) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,
                    steps=steps,  # Predict the next steps
                    rollout_steps=1,
                    truth=0,
                )
            elif args.task == "mujoco-v3":
                (
                    pred_states,
                    (avg_loss, trajectory_losses),
                ) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,  # Use the first 295 steps as context
                    steps=steps,  # Predict the next steps
                    rollout_steps=1,
                    truth=0,
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
    
    # Get all targets from the dataloader
    all_targets = torch.cat([targets for _, targets in test_loader], dim=0)
    all_targets = torch.roll(all_targets, shifts=1, dims=1)  # shift ground truth by 1
    
    print(
        "Saved ground truth shape:",
        all_targets[:num_preds, -predicted_states.shape[1] :, :].shape,
    )
    print("Saved losses shape:", losses.shape)
    np.save(
        f"sssm_{args.controller}_{args.task}_predictions_ar.npy",
        predicted_states.cpu().numpy(),
    )
    np.save(
        f"sssm_{args.controller}_{args.task}_ground_truths_ar.npy",
        all_targets[:num_preds, -predicted_states.shape[1] :, :].cpu().numpy(),
    )
    np.save(f"sssm_{args.controller}_{args.task}_losses_ar.npy", losses.cpu().numpy())
    print(
        f"Predictions, ground truths, and losses saved to 'sssm_{args.controller}_{args.task}_predictions.npy', 'sssm_{args.controller}_{args.task}_ground_truths.npy', and 'sssm_{args.controller}_{args.task}_losses.npy' respectively."
    )

    # # Plotting
    # plt.style.use("seaborn-v0_8-whitegrid")
    # sns.set_context("paper", font_scale=1.5)
    # colors = plt.cm.viridis(np.linspace(0, 1, num_preds))

    # fig = plt.figure(figsize=(20, 8 * num_preds))
    # gs = GridSpec(
    #     num_preds, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.3, hspace=0.4
    # )

    # for pred_idx in range(num_preds):
    #     print(f"Plotting prediction {pred_idx + 1}")

    #     # Plot predicted states vs ground truth
    #     ax1 = fig.add_subplot(gs[pred_idx, 0])
    #     feature_idx = 0

    #     # Plot ground truth
    #     ax1.plot(
    #         range(init, init + steps),
    #         test_targets[pred_idx, init : init + steps, feature_idx].cpu().numpy(),
    #         label="Ground Truth",
    #         color="black",
    #         linewidth=2,
    #         linestyle="--",
    #     )

    #     # Plot prediction
    #     ax1.plot(
    #         range(init, init + steps),
    #         predicted_states[pred_idx, : steps, feature_idx].cpu().numpy(),
    #         label="Predicted",
    #         color=colors[pred_idx],
    #         linewidth=2,
    #     )

    #     ax1.set_title(f"Prediction {pred_idx+1}: Predicted vs Ground Truth")
    #     ax1.set_xlabel("Time Step")
    #     ax1.set_ylabel("State Value")
    #     ax1.legend()

    #     # Plot losses
    #     ax2 = fig.add_subplot(gs[pred_idx, 1])

    #     ax2.plot(
    #         range(steps),
    #         smooth_curve(losses[pred_idx, : steps].cpu().numpy()),
    #         label="Total Loss",
    #         color="black",
    #         linewidth=2,
    #     )

    #     ax2.set_title(f"Prediction {pred_idx+1}: Losses")
    #     ax2.set_xlabel("Time Step")
    #     ax2.set_ylabel("Value")
    #     ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    #     ax2.set_yscale("log")  # Use log scale for better visibility

    # plt.suptitle(
    #     f"Spectral SSM Predictions for {args.controller} on {args.task}\n",
    #     fontsize=16,
    # )
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # # TODO: Add existok / make if non-existent (results/) directory
    # plt.savefig(
    #     f"results/sssm_{args.controller}_{args.task}_predictions.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.show()


if __name__ == "__main__":
    main()
