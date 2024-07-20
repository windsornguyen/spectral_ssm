# =============================================================================#
# Authors: Isabel Liu
# File: plot_predict.py
# =============================================================================#

"""Plotting script for Spectral SSM v. Transformer v. Mamba-2 v. Spectral Hybrid sequence prediction."""

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
parser.add_argument(
    "--feature",
    type=str,
    default="coordinates",
    choices=["coordinates", "orientations", "angles", "coordinate_velocities", "angular_velocities"],
    help="Features to plot the trajectory across timesteps."
)
args = parser.parse_args()

def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    return np.load(filename)

# Load data
try:
    sssm = load_data(f"sssm_{args.controller}_{args.task}_predictions_ar.npy")
    transformer = load_data(f"transformer_{args.controller}_{args.task}_predictions_ar.npy")
    mamba = load_data(f"mamba_{args.controller}_{args.task}_predictions_ar.npy")
    hybrid = load_data(f"hybrid_{args.controller}_{args.task}_predictions_ar.npy")

    ground_truth = load_data(f"sssm_{args.controller}_{args.task}_ground_truths_ar.npy")
    transformer_ground_truth = load_data(f"transformer_{args.controller}_{args.task}_ground_truths_ar.npy")
    mamba_ground_truth = load_data(f"mamba_{args.controller}_{args.task}_ground_truths_ar.npy")
    hybrid_ground_truth = load_data(f"hybrid_{args.controller}_{args.task}_ground_truths_ar.npy")

    sssm_losses = load_data(f"sssm_{args.controller}_{args.task}_losses_ar.npy")
    transformer_losses = load_data(f"transformer_{args.controller}_{args.task}_losses_ar.npy")
    mamba_losses = load_data(f"mamba_{args.controller}_{args.task}_losses_ar.npy")
    hybrid_losses = load_data(f"hybrid_{args.controller}_{args.task}_losses_ar.npy")
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

# Choose features to plot the predicted states (embeddings) v. ground truth states
if args.task in ["mujoco-v1", "mujoco-v2"]:
    if args.controller == "Ant-v1":
        if args.feature == "coordinates":
            feature_start, feature_end = 0, 3
        elif args.feature == "orientations":
            feature_start, feature_end = 3, 7
        elif args.feature == "angles":
            feature_start, feature_end = 7, 15
        elif args.feature == "coordinate_velocities":
            feature_start, feature_end = 15, 21
        else:
            feature_start, feature_end = 21, 29
    elif args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
        if args.feature == "coordinates":
            feature_start, feature_end = 0, 2
        elif args.feature == "angles":
            feature_start, feature_end = 2, 9
        elif args.feature == "coordinate_velocities":
            feature_start, feature_end = 9, 11
        else:
            feature_start, feature_end = 11, 18
    n_components = feature_end - feature_start
elif args.task == "mujoco-v3":
    n_components = 3  # to perform PCA
else:
    raise ValueError("Invalid task")

# Plotting
colors = ["b", "g", "r", "c", "m"]  # blue for Spectral SSM, green for Transformer, red for Ground Truth, cyan for Mamba, magenta for Hybrid
for pred_idx in range(num_preds):
    # Compute and plot the mean losses
    print(f"Plotting mean loss for prediction {pred_idx + 1}")

    # One figure for each prediction
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        range(time_steps),
        sssm_losses[pred_idx, : time_steps],
        label=f"Prediction {pred_idx+1} Spectral SSM Mean Loss",
        color=colors[0],
        linewidth=2,
    )
    ax.plot(
        range(time_steps),
        transformer_losses[pred_idx, : time_steps],
        label=f"Prediction {pred_idx+1} Transformer Mean Loss",
        color=colors[1],
        linewidth=2,
    )
    ax.plot(
        range(time_steps),
        mamba_losses[pred_idx, : time_steps],
        label=f"Prediction {pred_idx+1} Mamba Mean Loss",
        color=colors[3],
        linewidth=2,
    )
    ax.plot(
        range(time_steps),
        hybrid_losses[pred_idx, : time_steps],
        label=f"Prediction {pred_idx+1} Hybrid Mean Loss",
        color=colors[4],
        linewidth=2,
    )

    ax.legend()
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Mean Loss for Prediction {pred_idx+1}")
    # Save the mean loss figures
    plt.savefig(f"plots/predict_losses/mean_losses_preds_{pred_idx+1}_{args.controller}_{args.task}_ar.png")
    plt.close(fig)

# Perform PCA
if args.task == "mujoco-v3":
    pca = PCA(n_components=n_components)
    # Reshape: (num_preds, time_steps, num_features) -> (num_preds * time_steps, num_features) -> (num_preds, time_steps, n_components)
    sssm = pca.fit_transform(sssm.reshape(-1, num_features)).reshape(
        num_preds, time_steps, -1
    )
    transformer = pca.transform(transformer.reshape(-1, num_features)).reshape(
        num_preds, time_steps, -1
    )
    mamba = pca.transform(mamba.reshape(-1, num_features)).reshape(
        num_preds, time_steps, -1
    )
    hybrid = pca.transform(hybrid.reshape(-1, num_features)).reshape(
        num_preds, time_steps, -1
    )
    ground_truth = pca.transform(ground_truth.reshape(-1, num_features)).reshape(
        num_preds, time_steps, -1
    )

for pred_idx in range(num_preds):
    print(f"Plotting prediction {pred_idx + 1} over {time_steps} time steps")
    # One figure for each prediction
    fig, axs = plt.subplots(n_components, 1, figsize=(15, 5 * n_components))

    # Plot the predicted states (embeddings) and ground truth states
    for feature_idx in range(feature_start, feature_end):
        i = feature_idx - feature_start
        axs[i].plot(
            range(time_steps),
            ground_truth[pred_idx, : time_steps, feature_idx],
            label="Ground Truth",
            color=colors[2],
            linewidth=2,
            linestyle="--",
        )
        axs[i].plot(
            range(time_steps),
            sssm[pred_idx, : time_steps, feature_idx],
            label="Spectral SSM",
            color=colors[0],
            linewidth=2,
        )
        axs[i].plot(
            range(time_steps),
            transformer[pred_idx, : time_steps, feature_idx],
            label="Transformer",
            color=colors[1],
            linewidth=2,
        )
        axs[i].plot(
            range(time_steps),
            mamba[pred_idx, : time_steps, feature_idx],
            label="Mamba",
            color=colors[3],
            linewidth=2,
        )
        axs[i].plot(
            range(time_steps),
            hybrid[pred_idx, : time_steps, feature_idx],
            label="Hybrid",
            color=colors[4],
            linewidth=2,
        )
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel(f"Feature {feature_idx+1} Value")
        axs[i].set_title(
            f"Prediction {pred_idx+1}, Feature {feature_idx+1}"
        )
        axs[i].legend()

    plt.suptitle(
        f"Predictions for {args.controller} on {args.task}", fontsize=16
    )
    plt.tight_layout()

    # Save the figures
    plt.savefig(f"plots/preds/preds_{pred_idx+1}_{args.controller}_{args.task}_{args.feature}_ar.png")
    plt.show()
    plt.close(fig)
