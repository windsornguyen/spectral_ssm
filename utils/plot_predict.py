# =============================================================================#
# Authors: Isabel Liu
# File: plot_predict.py
# =============================================================================#

"""Plotting script for STU v. Transformer sequence prediction."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

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


# Load data
sssm = np.load("ssm_predictions_6l_huber.npy")
transformer = np.load("transformer_predictions_6l.npy")

ground_truth = np.load("ssm_ground_truths_6l_huber.npy")
transformer_ground_truth = np.load("transformer_ground_truths_6l.npy")

# Check if they are the same to ensure we are comparing the right data
if np.array_equal(ground_truth, transformer_ground_truth):
    print("The ground truth arrays are the same.")
else:
    print("The ground truth arrays are not the same.")

num_preds, time_steps, num_features = sssm.shape

# Assert if they are not of the same shape
assert (
    sssm.shape == transformer.shape == ground_truth.shape
), "The shapes of SSSM, Transformer, and Ground Truth are not the same."


# Choose features to plot the predicted states (embeddings) v. ground truth states
if args.task in ["mujoco-v1", "mujoco-v2"]:
    if args.controller == "Ant-v1":
        n_components = 3    # Ant-v1 coordinate features (in order): x-, y-, z-
    if args.controller in ["HalfCheetah-v1", "Walker2D-v1"]:
        n_components = 2    # HalfCheetah-v1/Walker2D-v1 coordinate features (in order): x-, z-
elif args.task == "mujoco-v3":
    n_components = 3 # to perform PCA
else:
    raise ValueError("Invalid task")

# Plotting
colors = ["b", "g", "r"]  # blue for SSSM, green for Transformer, red for Ground Truth
for pred_idx in range(num_preds):
    # Compute and plot the mean losses
    print(f"Plotting mean loss for prediction {pred_idx + 1}")

    # One figure for each prediction
    fig, ax = plt.subplots(figsize=(7, 5))

    sssm_mean_loss = np.mean(np.abs(ground_truth[pred_idx] - sssm[pred_idx]), axis=1)
    transformer_mean_loss = np.mean(np.abs(ground_truth[pred_idx] - transformer[pred_idx]), axis=1)

    ax.plot(
        range(time_steps),
        sssm_mean_loss,
        label=f"Prediction {pred_idx+1} SSSM Mean Loss",
        color=colors[0],
        linewidth=2,
    )
    ax.plot(
        range(time_steps),
        transformer_mean_loss,
        label=f"Prediction {pred_idx+1} Transformer Mean Loss",
        color=colors[1],
        linewidth=2,
    )
    ax.legend()
    # Save the mean loss figures
    plt.savefig(f"Mean_Losses_Prediction_{pred_idx+1}_6l_huber.png")
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
    fig, axs = plt.subplots(n_components, 2, figsize=(15, 5 * n_components))

    # Plot the predicted states (embeddings) and ground truth states
    for feature_idx in range(n_components):
        axs[feature_idx, 0].plot(
            range(time_steps),
            ground_truth[pred_idx, :time_steps, feature_idx],
            label=f"Prediction {pred_idx+1} Ground Truth, Feature {feature_idx+1}",
            color=colors[2],
            linewidth=2,
            linestyle="--",
        )
        axs[feature_idx, 0].plot(
            range(time_steps),
            sssm[pred_idx, :, feature_idx],
            label=f"Prediction {pred_idx+1} SSSM, Feature {feature_idx+1}",
            color=colors[0],
            linewidth=2,
        )
        axs[feature_idx, 0].plot(
            range(time_steps),
            transformer[pred_idx, :, feature_idx],
            label=f"Prediction {pred_idx+1} Transformer, Feature {feature_idx+1}",
            color=colors[1],
            linewidth=2,
        )

        # Plot the Feature losses
        sssm_loss = np.abs(ground_truth[pred_idx, :, feature_idx] - sssm[pred_idx, :, feature_idx])
        transformer_loss = np.abs(ground_truth[pred_idx, :, feature_idx] - transformer[pred_idx, :, feature_idx])

        axs[feature_idx, 1].plot(
            range(time_steps),
            sssm_loss,
            label=f"Prediction {pred_idx+1} SSSM Loss, Feature {feature_idx+1}",
            color=colors[0],
            linewidth=2,
        )
        axs[feature_idx, 1].plot(
            range(time_steps),
            transformer_loss,
            label=f"Prediction {pred_idx+1} Transformer Loss, Feature {feature_idx+1}",
            color=colors[1],
            linewidth=2,
        )

    # Show the plots
    for ax in axs.flat:
        ax.legend()
    
    # Save the figures
    plt.savefig(f"Predictions_{pred_idx+1}_6l_huber.png")
    plt.close(fig)
