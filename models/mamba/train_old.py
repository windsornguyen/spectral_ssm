import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from stu.physics.physics_data import get_dataloader
from mamba.model import Mamba2, Mamba2Config, InferenceParameters
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss

import torch.distributed as dist
from scipy.ndimage import gaussian_filter1d
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)


def plot_losses(losses, title, eval_interval=None, ylabel='Loss'):
    if eval_interval:
        x_values = [i * eval_interval for i in range(len(losses))]
    else:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses), label=title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()


@torch.no_grad()
def evaluate(model, loader, inference_params):
    model.eval()
    device = next(model.parameters()).device
    losses = []

    for X, y in tqdm(loader, desc='Evaluating', unit='iter'):
        X, y = X.to(device), y.to(device)
        preds, loss = model(inputs=X, targets=y, inference_params=inference_params)
        loss, _ = loss
        losses.append(loss.item())

    model.train()
    return np.mean(losses)


def main():
    # Set seeds for reproducibility
    torch.manual_seed(1337)
    np.random.seed(1337)

    # Hyperparameters
    batch_size = 3
    max_len = 1_000
    num_epochs = 3
    eval_interval = 100
    lr = 7.5e-4
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device:', device)
    d_model = 24 # TODO: Needs to be divisible by 8 otherwise custom conv1d lib won't work
    d_out = 18
    d_state = 32
    d_conv = 4
    expand = 2
    headdim = 1
    d_ssm = None
    ngroups = 1
    A_init_range = (1, 16)
    D_has_hdim = False
    rmsnorm = True
    norm_before_gate = False
    dt_min = 0.001
    dt_max = 0.1
    dt_init_floor = 1e-4
    dt_limit = (0.0, float('inf'))
    bias = False
    conv_bias = True
    chunk_size = 64
    use_mem_eff_path = True
    layer_idx = 0
    dropout = 0.25 # TODO: Currently not used
    patience = 5

    # Data loading
    controller = 'HalfCheetah-v1'
    train_inputs = f'../data/{controller}/3000/train_inputs.npy'
    train_targets = f'../data/{controller}/3000/train_targets.npy'
    val_inputs = f'../data/{controller}/3000/val_inputs.npy'
    val_targets = f'../data/{controller}/3000/val_targets.npy'
    print(f'Training on {controller} task.')

    # Get dataloaders
    train_loader = get_dataloader(train_inputs, train_targets, batch_size, device)
    val_loader = get_dataloader(val_inputs, val_targets, batch_size, device)

    # Set the loss function based on the controller
    loss_fn = HalfCheetahLoss() if controller == 'HalfCheetah-v1' else Walker2DLoss() if controller == 'Walker2D-v1' else AntLoss()

    configs = {
        'd_model': d_model,
        'd_state': d_state,
        'd_conv': d_conv,
        'expand': expand,
        'headdim': headdim,
        'd_ssm': d_ssm,
        'ngroups': ngroups,
        'A_init_range': A_init_range,
        'D_has_hdim': D_has_hdim,
        'rmsnorm': rmsnorm,
        'norm_before_gate': norm_before_gate,
        'dt_min': dt_min,
        'dt_max': dt_max,
        'dt_init_floor': dt_init_floor,
        'dt_limit': dt_limit,
        'bias': bias,
        'conv_bias': conv_bias,
        'chunk_size': chunk_size,
        'use_mem_eff_path': use_mem_eff_path,
        'layer_idx': layer_idx,
        'dropout': dropout,
        'loss_fn': loss_fn,
        'max_len': max_len,
        'd_out': d_out,
        'device': device
    }

    configs = Mamba2Config(**configs)
    inference_params = InferenceParameters()
    model = Mamba2(configs)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    grad_norms = []

    # Check available individual losses once before the training loop
    metric_losses = {
        'coordinate_loss': [],
        'orientation_loss': [],
        'angle_loss': [],
        'coordinate_velocity_loss': [],
        'angular_velocity_loss': []
    }

    pbar = tqdm(range(num_epochs * len(train_loader)), desc='Training', unit='iter')
    for epoch in range(num_epochs):
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            # Evaluate the loss
            preds, loss = model(inputs=xb, targets=yb, inference_params=inference_params)
            loss, _ = loss
            train_losses.append(loss)

            # Check if metric exists first for the given loss function
            if hasattr(loss_fn, 'metrics'):
                metrics = loss_fn.metrics(preds, yb)
                for metric in metric_losses:
                    if metric in metrics:
                        metric_losses[metric].append(metrics[metric])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Print gradients to check for NaN and compute grad norm
            grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    if torch.isnan(param.grad).any():
                        print(f"NaN gradient in {name}")
            grad_norm = grad_norm ** 0.5
            grad_norms.append(grad_norm)

            optimizer.step()

            # Evaluate on validation set
            total_steps = epoch * len(train_loader) + step
            if (total_steps % eval_interval == 0) or total_steps == num_epochs * len(train_loader) - 1:
                val_loss = evaluate(model, val_loader, inference_params)
                val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'best_{controller}_mamba2.safetensors')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f'Early stopping triggered. Best validation loss: {best_val_loss:.4f}')
                    break

            postfix_dict = {
                'tr_loss': loss.item(),
                'val_loss': val_losses[-1] if len(val_losses) > 0 else None,
                'grd_nrm': grad_norm
            }
            if hasattr(loss_fn, 'metrics'):
                for metric in metrics:
                    postfix_dict[metric] = metrics[metric]
            
            pbar.set_postfix(postfix_dict)
            pbar.update(1)

    plt.style.use('seaborn-v0_8-whitegrid')
    if not os.path.exists('results'):
        os.makedirs('results')

    # Plot training and validation losses (main losses - losses.png)
    plt.figure(figsize=(10, 5))
    plot_losses(train_losses, 'Training Loss')
    plot_losses(val_losses, 'Validation Loss', eval_interval)
    plt.title(f'Training and Validation Losses on {controller} Task (Mamba2)')
    plt.tight_layout()
    plt.savefig(f'results/{controller}_mamba2_losses.png', dpi=300)
    plt.show()
    plt.close()

    # Plot other losses and gradient norm (other losses - details.png)
    plt.figure(figsize=(10, 5))
    for metric, losses in metric_losses.items():
        plot_losses(losses, metric)
    plot_losses(grad_norms, 'Gradient Norm', ylabel='Gradient Norm')
    plt.title(f'Other Losses, Gradient Norm Over Time on {controller} Task (Mamba2)')
    plt.tight_layout()
    plt.savefig(f'results/{controller}_mamba2_details.png', dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
