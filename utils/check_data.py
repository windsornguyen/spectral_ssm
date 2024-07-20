import numpy as np

# Load data
inputs = np.load("data/mujoco-v1/Walker2D-v1/train_inputs.npy")
targets = np.load("data/mujoco-v1/Walker2D-v1/train_targets.npy")

# Initialize an empty array for shifted inputs
inputs = inputs[:, :, :18]
inputs_shifted = np.empty_like(inputs)

# Loop over each sequence in the batch
for i in range(inputs.shape[0]):
    # Shift the sequence forward by one step
    inputs_shifted[i] = np.roll(inputs[i], shift=-1, axis=0)

# Remove the first entry from both logs to make them the same length
inputs_shifted = inputs_shifted[:, :-1, :]
targets = targets[:, :-1, :]

# Use numpy.allclose to check if targets and the shifted inputs are the same
if np.allclose(inputs_shifted, targets):
    print("The shifted inputs and targets are the same.")
else:
    print("The shifted inputs and targets are not the same.")

print(f"inputs: {inputs[0, :5, :3]}")
print(f"targets: {targets[0, :5, :3]}")
print(f"shifted inputs: {inputs_shifted[0, :5, :3]}")