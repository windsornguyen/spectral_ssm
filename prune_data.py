import numpy as np

# Load the .npy file
data = np.load('data/mujoco-v1/Walker2D-v1/train_targets_orig.npy')

# Select the first 29 features
data_new = data[:, :900, :]

print(data_new.shape)

# Save the new data to a .npy file
np.save('data/mujoco-v1/Walker2D-v1/train_targets.npy', data_new)