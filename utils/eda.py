import numpy as np
from sklearn.model_selection import train_test_split

# Load the input and target data
# controller = "Walker2D-v1"
input_file = "data/mujoco-v1/Walker2D-v1/raw_inputs.npy"
target_file = "data/mujoco-v1/Walker2D-v1/raw_targets.npy"
# # Ant (in, out) dims: (37, 29)
# # Walker2D (in, out)  dims: (24, 18)
# # HalfCheetah (in, out)  dims: (24, 18)

input_data = np.load(input_file)
target_data = np.load(target_file)

# # Perform EDA
print("Input Data Shape:", input_data.shape)
print("Target Data Shape:", target_data.shape)
print("Input Data Sample:", input_data[:5])
print("Target Data Sample:", target_data[:5])

# # Plot some samples
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.hist(input_data.ravel(), bins=50)
# plt.title("Input Data Distribution")

# plt.subplot(1, 2, 2)
# plt.hist(target_data.ravel(), bins=50)
# plt.title("Target Data Distribution")

# plt.show()

# # Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    input_data, target_data, test_size=0.2, random_state=1337
)

print("Input Data Stats:")
print("Means:", np.mean(input_data, axis=0))
print("Standard Deviations:", np.std(input_data, axis=0))
print("Min:", np.min(input_data, axis=0))
print("Max:", np.max(input_data, axis=0))

print("\nOutput Data Stats:")
print("Means:", np.mean(target_data, axis=0))
print("Standard Deviations:", np.std(target_data, axis=0))
print("Min:", np.min(target_data, axis=0))
print("Max:", np.max(target_data, axis=0))

# Print dimensions of the splts
print(f"Train inputs shape: {X_train.shape}, train targets shape: {y_train.shape}")
print(f"Val inputs shape: {X_val.shape}, val targets shape: {y_val.shape}")

# Save the split data to .npy files
np.save("data/mujoco-v1/Walker2D-v1/train_inputs.npy", X_train)
np.save("data/mujoco-v1/Walker2D-v1/train_targets.npy", y_train)
np.save("data/mujoco-v1/Walker2D-v1/val_inputs.npy", X_val)
np.save("data/mujoco-v1/Walker2D-v1/val_targets.npy", y_val)


# # Modified function to load data from the new .npy files
# def get_batch(split):
#     if split == "train":
#         inputs = np.load(os.path.join(data_dir, "train_inputs.npy"))
#         targets = np.load(os.path.join(data_dir, "train_targets.npy"))
#     else:
#         inputs = np.load(os.path.join(data_dir, "val_inputs.npy"))
#         targets = np.load(os.path.join(data_dir, "val_targets.npy"))

#     ix = np.random.randint(0, len(inputs) - ctxt_len, batch_size)
#     x = np.stack([inputs[i : i + ctxt_len] for i in ix])
#     y = np.stack([targets[i + 1 : i + 1 + ctxt_len] for i in ix])

#     x = torch.from_numpy(x).long()
#     y = torch.from_numpy(y).long()

#     if device == "cuda":
#         x, y = (
#             x.pin_memory().to(device, non_blocking=True),
#             y.pin_memory().to(device, non_blocking=True),
#         )
#     else:
#         x, y = x.to(device), y.to(device)

#     return x, y
