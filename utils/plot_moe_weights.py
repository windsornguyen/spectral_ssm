# =============================================================================#
# Authors: Isabel Liu
# File: plot_moe_weights.py
# =============================================================================#

"""Plotting script for MoE-STU weights."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Process command line flags
parser = argparse.ArgumentParser(
    description="Plotting script for MoE-STU weights"
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

# Define base path and file names
base_path = f'results/{args.task}/sssm/sssm-{args.controller}'
weights_file = f'{base_path}-weights.npy'
plot_file = f'{base_path}-weights_plot.png'

def plot_moe_weights(weights_history, save_path, window_size=50):
    # window_size is the size of the moving window for the moving average (i.e. amount of smoothing)
    weights_history = pd.DataFrame(weights_history)
    weights_history = weights_history.rolling(window_size).mean()

    plt.figure(figsize=(10, 6))
    for i in range(weights_history.shape[1]):
        plt.plot(weights_history.iloc[:, i], label=f'STU {i+1}')
    plt.title('Weights of STUs over time')
    plt.xlabel('Step')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    weights_history = np.load(weights_file)
    plot_moe_weights(weights_history, plot_file)
