# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: plot_predict.py
# =============================================================================#

"""Plotting script for spectral SSM v. Transformer sequence prediction."""

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
    default="mujoco-v3",
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
    transformer = load_data(
        f"transformer_{args.controller}_{args.task}_predictions.npy"
    )
    ground_truth = load_data(f"sssm_{args.controller}_{args.task}_ground_truths.npy")
    transformer_ground_truth = load_data(
        f"transformer_{args.controller}_{args.task}_ground_truths.npy"
    )
    sssm_losses = load_data(f"sssm_{args.controller}_{args.task}_losses.npy")
    transformer_losses = load_data(f"transformer_{args.controller}_{args.task}_losses.npy")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

# Check if they are the same to ensure we are comparing the right data
if np.array_equal(ground_truth, transformer_ground_truth):
    print("The ground truth arrays are the same.")
else:
    print("The ground truth arrays are not the same.")

num_preds, time_steps, num_features = sssm.shape

# Assert if they are not of the same shape
print(sssm.shape, transformer.shape, ground_truth.shape)
assert (
    sssm.shape == transformer.shape == ground_truth.shape
), "The shapes of Spectral SSM, Transformer, and Ground Truths are not the same."

# Choose features to plot the predicted states (embeddings) v. ground truth states
if args.task in ["mujoco-v1", "mujoco-v2"]:
    if args.controller == "Ant-v1":
        n_components = 3  # Ant-v1 coordinate features (in order): x-, y-, z-
    if args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
        n_components = (
            2  # HalfCheetah-v1/Walker2D-v1 coordinate features (in order): x-, z-
        )
elif args.task == "mujoco-v3":
    n_components = 3  # to perform PCA
else:
    raise ValueError("Invalid task")

# Plotting
colors = ["b", "g", "r"]  # blue for Spectral SSM, green for Transformer, red for Ground Truth
for pred_idx in range(num_preds):
    # Compute and plot the mean losses
    print(f"Plotting mean loss for prediction {pred_idx + 1}")

    # One figure for each prediction
    fig, ax = plt.subplots(figsize=(7, 5))

    # sssm_mean_loss = np.mean(np.abs(ground_truth[pred_idx] - sssm[pred_idx]), axis=1)
    # transformer_mean_loss = np.mean(
    #     np.abs(ground_truth[pred_idx] - transformer[pred_idx]), axis=1
    # )

    ax.plot(
        range(time_steps - 1),
        sssm_losses[pred_idx, : time_steps - 1],
        label=f"Prediction {pred_idx+1} Spectral SSM Mean Loss",
        color=colors[0],
        linewidth=2,
    )
    ax.plot(
        range(time_steps - 1),
        transformer_losses[pred_idx, : time_steps - 1],
        label=f"Prediction {pred_idx+1} Transformer Mean Loss",
        color=colors[1],
        linewidth=2,
    )

    ax.legend()
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Mean Loss for Prediction {pred_idx+1}")
    # Save the mean loss figures
    plt.savefig(f"plots/losses/mean_losses_preds_{pred_idx+1}_{args.controller}_{args.task}.png")
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
    ground_truth = pca.transform(ground_truth.reshape(-1, num_features)).reshape(
        num_preds, time_steps, -1
    )

for pred_idx in range(num_preds):
    print(f"Plotting prediction {pred_idx + 1} over {time_steps} time steps")
    # One figure for each prediction
    fig, axs = plt.subplots(n_components, 1, figsize=(15, 5 * n_components))

    # Plot the predicted states (embeddings) and ground truth states
    for feature_idx in range(n_components):
        axs[feature_idx].plot(
            range(time_steps - 1),
            ground_truth[pred_idx, : time_steps - 1, feature_idx],
            label="Ground Truth",
            color=colors[2],
            linewidth=2,
            linestyle="--",
        )
        axs[feature_idx].plot(
            range(time_steps - 1),
            sssm[pred_idx, : time_steps - 1, feature_idx],
            label="Spectral SSM",
            color=colors[0],
            linewidth=2,
        )
        axs[feature_idx].plot(
            range(time_steps - 1),
            transformer[pred_idx, : time_steps - 1, feature_idx],
            label="Transformer",
            color=colors[1],
            linewidth=2,
        )
        axs[feature_idx].set_xlabel("Time Step")
        axs[feature_idx].set_ylabel(f"Feature {feature_idx+1} Value")
        axs[feature_idx].set_title(
            f"Prediction {pred_idx+1}, Feature {feature_idx+1}"
        )
        axs[feature_idx].legend()

        # # Plot the Feature losses
        # sssm_loss = np.abs(
        #     ground_truth[pred_idx, :, feature_idx] - sssm[pred_idx, :, feature_idx]
        # )
        # transformer_loss = np.abs(
        #     ground_truth[pred_idx, :, feature_idx]
        #     - transformer[pred_idx, :, feature_idx]
        # )

    plt.suptitle(
        f"Predictions for {args.controller} on {args.task}", fontsize=16
    )
    plt.tight_layout()

    # Save the figures
    plt.savefig(f"plots/preds/preds_{pred_idx+1}_{args.controller}_{args.task}.png")
    plt.show()
    plt.close(fig)
