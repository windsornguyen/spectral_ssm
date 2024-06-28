import torch
import numpy as np
import matplotlib.pyplot as plt
from spectral_ssm import model
import safetensors

# TODO: Clean up the locations of various predict.py files.
def main():
    # Load the trained model
    model_path = 'checkpoints/best_model.safetensors'
    model_args = {
        'd_out': 37,
        'input_len': 1000,
        'num_eigh': 24,
        'auto_reg_k_u': 3,
        'auto_reg_k_y': 2,
        'learnable_m_y': True,
    }
    m = model.STU(**model_args)
    state_dict = safetensors.load_file(model_path)
    m.load_state_dict(state_dict)
    m.eval()

    # Load the test data
    test_inputs = 'data/Ant-v1/test_inputs.npy'
    test_targets = 'data/Ant-v1/test_targets.npy'
    test_inputs = torch.tensor(np.load(test_inputs), dtype=torch.float32)
    test_targets = torch.tensor(np.load(test_targets), dtype=torch.float32)
    # Print dims of inputs and targets
    print(test_inputs.shape, test_targets.shape)

    # Select a specific slice of trajectories
    seq_idx = 0
    input_trajectories = test_inputs[seq_idx:seq_idx+5]  # Select 5 input trajectories starting from seq_idx
    target_trajectories = test_targets[seq_idx:seq_idx+5]  # Select 5 target trajectories starting from seq_idx
    # Print dims of inputs and targets
    print(input_trajectories.shape, target_trajectories.shape)

    # Predict the next states using the model
    init_idx = 0
    t = 100  # Number of time steps to predict
    predicted_states, losses = m.predict(input_trajectories, target_trajectories, init=init_idx, t=t)

    # Extract the individual losses from the loss tuple
    total_loss, metrics = losses
    losses = metrics['loss']
    coordinate_loss = metrics['coordinate_loss']
    orientation_loss = metrics['orientation_loss']
    angle_loss = metrics['angle_loss']
    coordinate_velocity_loss = metrics['coordinate_velocity_loss']
    angular_velocity_loss = metrics['angular_velocity_loss']

    # Plot the predicted states and ground truth states
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(init_idx, init_idx + t), target_trajectories[0, init_idx:init_idx + t, 0], label='Ground Truth')
    ax.plot(range(init_idx, init_idx + t), [state[0][0] for state in predicted_states], label='Predicted')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State')
    ax.set_title('Predicted vs Ground Truth States')
    ax.legend()

    # Print the averaged total loss and plot losses over different time steps
    losses = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in losses]
    print("Averaged Total Loss:", total_loss)
    fig2, axz = plt.subplots(figsize=(10, 6))
    axz.plot(range(init_idx, init_idx + t), losses)
    axz.set_title('Total Loss')

    # Plot the individual losses
    coordinate_loss = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in coordinate_loss]
    orientation_loss = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in orientation_loss]
    angle_loss = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in angle_loss]
    coordinate_velocity_loss = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in coordinate_velocity_loss]
    angular_velocity_loss = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in angular_velocity_loss]

    fig3, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0, 0].plot(range(init_idx, init_idx + t), coordinate_loss)
    axs[0, 0].set_title('Coordinate Loss')
    axs[0, 1].plot(range(init_idx, init_idx + t), orientation_loss)
    axs[0, 1].set_title('Orientation Loss')
    axs[0, 2].plot(range(init_idx, init_idx + t), angle_loss)
    axs[0, 2].set_title('Angle Loss')
    axs[1, 0].plot(range(init_idx, init_idx + t), coordinate_velocity_loss)
    axs[1, 0].set_title('Coordinate Velocity Loss')
    axs[1, 1].plot(range(init_idx, init_idx + t), angular_velocity_loss)
    axs[1, 1].set_title('Angular Velocity Loss')
    axs[1, 2].axis('off')  # Leave the last subplot empty

    for ax in axs.flat:
        ax.set(xlabel='Time Step', ylabel='Loss')

    fig3.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
