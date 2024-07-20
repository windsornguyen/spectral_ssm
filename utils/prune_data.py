import numpy as np

# Load the .npy file
data = np.load("data/mujoco-v1/Walker2D-v1/raw_inputs.npy")

# Select the first 29 features
data_new = data[:, :, :18]

print(data_new.shape)

# Save the new data to a .npy file
np.save("data/mujoco-v2/Walker2D-v1/raw_inputs.npy", data_new)