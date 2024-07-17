# =============================================================================#
# Authors: Isabel Liu
# File: plot_predict_mean_loss.py
# =============================================================================#

"""Plotting the mean losses for Spectral SSM v. Transformer v. Mamba-2 v. Spectral Hybrid sequence prediction."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import os

# Process command line flags
parser = argparse.ArgumentParser(
    description="Plotting script for sequence prediction inference"
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
    default="mujoco-v1",
    choices=["mujoco-v1", "mujoco-v2", "mujoco-v3"],
    help="Task to run inference on.",
)
args = parser.parse_args()

def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    return np.load(filename)

# Load data
try:
    sssm = load_data(f"sssm_{args.controller}_{args.task}_predictions.npy")
    transformer = load_data(f"transformer_{args.controller}_{args.task}_predictions.npy")
    mamba = load_data(f"mamba_{args.controller}_{args.task}_predictions.npy")
    hybrid = load_data(f"hybrid_{args.controller}_{args.task}_predictions.npy")

    ground_truth = load_data(f"sssm_{args.controller}_{args.task}_ground_truths.npy")
    transformer_ground_truth = load_data(f"transformer_{args.controller}_{args.task}_ground_truths.npy")
    mamba_ground_truth = load_data(f"mamba_{args.controller}_{args.task}_ground_truths.npy")
    hybrid_ground_truth = load_data(f"hybrid_{args.controller}_{args.task}_ground_truths.npy")

    sssm_losses = load_data(f"sssm_{args.controller}_{args.task}_losses.npy")
    transformer_losses = load_data(f"transformer_{args.controller}_{args.task}_losses.npy")
    mamba_losses = load_data(f"mamba_{args.controller}_{args.task}_losses.npy")
    hybrid_losses = load_data(f"hybrid_{args.controller}_{args.task}_losses.npy")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

if args.controller == "Ant-v1":
    # Remove zero-padding for Mamba-2
    mamba = mamba[:, :, :-3]
    mamba_ground_truth = mamba_ground_truth[:, :, :-3]

# Check if they are the same to ensure we are comparing the right data
if np.array_equal(ground_truth, transformer_ground_truth) and np.array_equal(ground_truth, mamba_ground_truth) and np.array_equal(ground_truth, hybrid_ground_truth):
    print("The ground truth arrays are the same.")
else:
    print("The ground truth arrays are not the same.")

num_preds, time_steps, num_features = sssm.shape

# Assert if they are not of the same shape
print(sssm.shape, transformer.shape, mamba.shape, hybrid.shape, ground_truth.shape)
assert (
    sssm.shape == transformer.shape == mamba.shape == hybrid.shape == ground_truth.shape
), "The shapes of Spectral SSM, Transformer, Mamba, Hybrid, and Ground Truths are not the same."

# Compute the mean across all trajectories
mean_sssm = sssm.mean(axis=0)
mean_transformer = transformer.mean(axis=0)
mean_mamba = mamba.mean(axis=0)
mean_hybrid = hybrid.mean(axis=0)

mean_ground_truth = ground_truth.mean(axis=0)

mean_sssm_losses = sssm_losses.mean(axis=0)
mean_transformer_losses = transformer_losses.mean(axis=0)
mean_mamba_losses = mamba_losses.mean(axis=0)
mean_hybrid_losses = hybrid_losses.mean(axis=0)

# Plotting
colors = ["b", "g", "r", "c", "m"]  # blue for Spectral SSM, green for Transformer, red for Ground Truth, cyan for Mamba, magenta for Hybrid

# Compute and plot the mean losses
print(f"Plotting mean loss for averaged predictions")

# One figure for each prediction
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    range(time_steps),
    mean_sssm_losses[: time_steps],
    label=f"Averaged Spectral SSM Mean Loss",
    color=colors[0],
    linewidth=2,
)
ax.plot(
    range(time_steps),
    mean_transformer_losses[: time_steps],
    label=f"Averaged Transformer Mean Loss",
    color=colors[1],
    linewidth=2,
)
ax.plot(
    range(time_steps),
    mean_mamba_losses[: time_steps],
    label=f"Averaged Mamba Mean Loss",
    color=colors[3],
    linewidth=2,
)
ax.plot(
    range(time_steps),
    mean_hybrid_losses[: time_steps],
    label=f"Averaged Hybrid Mean Loss",
    color=colors[4],
    linewidth=2,
)

ax.legend()
ax.set_xlabel("Time Step")
ax.set_ylabel("Mean Absolute Error")
ax.set_title(f"Mean Loss for Averaged Predictions")
# Save the mean loss figures
plt.savefig(f"plots/predict_losses/mean_losses_averaged_{args.controller}_{args.task}.png")
plt.close(fig)
