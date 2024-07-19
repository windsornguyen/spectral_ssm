# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: predict.py
# =============================================================================#

"""Prediction loop for STU sequence prediction."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from safetensors.torch import load_file
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

from models.stu.model import SpectralSSM, SpectralSSMConfigs
from torch.nn import MSELoss
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from data_utils import get_dataloader

def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)

def plot_losses(losses, title, x_values=None, ylabel="Loss", color=None):
    if x_values is None:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses), label=title, color=color)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()

def load_test_data(args):
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        val_inputs = np.load(f"{base_path}/val_inputs.npy")
        val_targets = np.load(f"{base_path}/val_targets.npy")
        return {"inputs": val_inputs, "targets": val_targets}
    elif args.task == "mujoco-v3":
        return torch.load(f"data/{args.task}/{args.controller}/{args.controller}_ResNet-18_test.pt")
    else:
        raise ValueError("Invalid task")

def setup_model_configs(args, device):
    if args.task == "mujoco-v1":
        sl = 1000
        if args.controller == "Ant-v1":
            n_embd, d_out, d_proj = 37, 37, 29
            loss_fn = AntLoss()
        elif args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
            n_embd, d_out, d_proj = 24, 24, 18
            loss_fn = HalfCheetahLoss() if args.controller == "HalfCheetah-v1" else Walker2DLoss()
        else:
            raise ValueError("Invalid controller for mujoco-v1")
    elif args.task == "mujoco-v2":
        sl = 1000
        if args.controller == "Ant-v1":
            n_embd, d_out, d_proj = 29, 29, 29
            loss_fn = AntLoss()
        elif args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
            n_embd, d_out, d_proj = 18, 18, 18
            loss_fn = HalfCheetahLoss() if args.controller == "HalfCheetah-v1" else Walker2DLoss()
        else:
            raise ValueError("Invalid controller for mujoco-v2")
    elif args.task == "mujoco-v3":
        sl, n_embd, d_out, d_proj = 300, 512, 512, 512
        loss_fn = MSELoss()
    else:
        raise ValueError("Invalid task")

    return SpectralSSMConfigs(
        n_layers=2,
        n_embd=n_embd,
        d_in=n_embd,
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

def run_inference(model, test_loader, num_preds, args):
    predicted_states = []
    losses = []
    init = 950 if args.task in ["mujoco-v1", "mujoco-v2"] else 295
    steps = 50  # Assuming we want to predict 50 steps ahead

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= num_preds:
                break

            if args.task in ["mujoco-v1", "mujoco-v2"]:
                pred_states, (avg_loss, trajectory_losses) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,
                    steps=steps,
                    rollout_steps=20,
                )
            elif args.task == "mujoco-v3":
                pred_states, (avg_loss, trajectory_losses) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,
                    steps=steps,
                    rollout_steps=1,
                )

            predicted_states.append(pred_states)
            losses.append(trajectory_losses)

    return torch.cat(predicted_states, dim=0), torch.cat(losses, dim=0)

def save_results(predicted_states, test_loader, losses, args):
    np.save(f"sssm_{args.controller}_{args.task}_predictions.npy", predicted_states.cpu().numpy())
    np.save(f"sssm_{args.controller}_{args.task}_ground_truths.npy", next(iter(test_loader))[1].cpu().numpy())
    np.save(f"sssm_{args.controller}_{args.task}_losses.npy", losses.cpu().numpy())
    print(f"Results saved for {args.controller} on {args.task}")

def plot_results(predicted_states, test_loader, losses, args):
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.5)
    colors = plt.cm.viridis(np.linspace(0, 1, predicted_states.shape[0]))

    fig = plt.figure(figsize=(20, 8 * predicted_states.shape[0]))
    gs = GridSpec(predicted_states.shape[0], 2, figure=fig, width_ratios=[1, 1.2], wspace=0.3, hspace=0.4)

    init = 950 if args.task in ["mujoco-v1", "mujoco-v2"] else 295
    steps = predicted_states.shape[1]

    for pred_idx in range(predicted_states.shape[0]):
        print(f"Plotting prediction {pred_idx + 1}")

        # Plot predicted states vs ground truth
        ax1 = fig.add_subplot(gs[pred_idx, 0])
        feature_idx = 0

        # Plot ground truth
        ground_truth = next(iter(test_loader))[1][pred_idx]
        ax1.plot(
            range(init, init + steps),
            ground_truth[init : init + steps, feature_idx].cpu().numpy(),
            label="Ground Truth",
            color="black",
            linewidth=2,
            linestyle="--",
        )

        # Plot prediction
        ax1.plot(
            range(init, init + steps),
            predicted_states[pred_idx, :, feature_idx].cpu().numpy(),
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
            smooth_curve(losses[pred_idx, :].cpu().numpy()),
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
        f"Spectral SSM Predictions for {args.controller} on {args.task}\n",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    plt.savefig(
        f"results/sssm_{args.controller}_{args.task}_predictions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Inference script for sequence prediction")
    parser.add_argument("--controller", type=str, default="Ant-v1", choices=["Ant-v1", "HalfCheetah-v1", "Walker2D-v1"])
    parser.add_argument("--task", type=str, default="mujoco-v3", choices=["mujoco-v1", "mujoco-v2", "mujoco-v3"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shift", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up model configurations
    configs = setup_model_configs(args, device)

    # Initialize and load the model
    model = SpectralSSM(configs).to(device)
    state_dict = load_file(f"sssm_{args.controller}_{args.task}.pt", device=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test data
    test_data = load_test_data(args)

    # Get data loader
    test_loader = get_dataloader(
        model="spectral_ssm",
        data=test_data,
        task=args.task,
        controller=args.controller,
        bsz=args.batch_size,
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
    num_preds = 5
    predicted_states, losses = run_inference(model, test_loader, num_preds, args)

    # Save predictions, ground truths, and losses
    save_results(predicted_states, test_loader, losses, args)

    # Plotting
    plot_results(predicted_states, test_loader, losses, args)

if __name__ == "__main__":
    main()
