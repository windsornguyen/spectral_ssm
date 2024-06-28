# =============================================================================#
# Authors: Isabel Liu, Yagiz Devre, Windsor Nguyen
# File: train.py
# =============================================================================#

"""Training loop for physics sequence prediction."""

import os
import sys

# TODO: Fix annoying Python pkg path issues eventually so we can remove this.
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# TODO: Organize imports acording to PEP8 standards.
import argparse
import random
from datetime import datetime
from socket import gethostname

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from stu import experiment, model, optimizer
from stu.physics import physics_data
from stu.model import STUConfigs
from transformer.model import Transformer, TransformerConfigs
# from mamba.model import Mamba, MambaConfig
# from jamba.model import Jamba, JambaConfig
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(
    rank: int, world_size: int, gpus_per_node: int
) -> tuple[torch.device, int, int]:
    """
    Adapts to distributed or non-distributed training environments.
    Chooses appropriate backend and device based on the available hardware and environment setup.
    Manages NCCL for NVIDIA GPUs, Gloo for CPUs, and potentially Gloo for Apple Silicon (MPS).
    """
    local_rank = rank % gpus_per_node if gpus_per_node > 0 else 0
    device = torch.device('cpu')  # Default to CPU
    backend = 'gloo'  # Default backend

    if world_size > 1 and 'SLURM_PROCID' in os.environ:
        if torch.cuda.is_available() and gpus_per_node > 0:
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = 'nccl'
            dist.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )
            print(
                f'host: {gethostname()}, rank: {rank}, local_rank: {local_rank}'
            )
            if rank == 0:
                print(f'Group initialized? {dist.is_initialized()}', flush=True)
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            dist.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )
            print(f'Using MPS on host: {gethostname()}, rank: {rank}')
            if rank == 0:
                print(f'Group initialized? {dist.is_initialized()}', flush=True)
    else:
        # Non-distributed fallback to the best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')

    return device, local_rank, world_size


def gaussian_kernel(size, sigma):
    """Creates a 1D Gaussian kernel using PyTorch."""
    size = int(size) // 2
    x = torch.arange(-size, size + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_curve(points, sigma=2):
    """Applies 1D Gaussian smoothing on a list of points."""
    kernel_size = int(
        4 * sigma + 1
    )  # Kernel size, covering +/- 4 standard deviations
    points = torch.tensor(points, dtype=torch.float32)
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0)
    # Apply padding to handle borders
    points_padded = torch.nn.functional.pad(
        points, (kernel_size // 2, kernel_size // 2), mode='reflect'
    )
    smoothed_points = torch.nn.functional.conv1d(
        points_padded.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0)
    )
    return smoothed_points.squeeze().numpy()


def plot_losses(losses, title, eval_interval=None, ylabel='Loss'):
    """Plots smoothed loss curve."""
    if eval_interval:
        x_values = [i * eval_interval for i in range(len(losses))]
    else:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses, sigma=2), label=title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_metrics(
    train_losses,
    val_losses,
    metric_losses,
    grad_norms,
    output_dir,
    controller,
    eval_interval,
):
    plt.style.use('seaborn-v0_8-whitegrid')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot training and validation losses (main losses - losses.png)
    plt.figure(figsize=(10, 5))
    plot_losses(train_losses, 'Training Loss')
    plot_losses(val_losses, 'Validation Loss', eval_interval)
    plt.title(f'Training and Validation Losses on {controller} Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{controller}_losses.png'), dpi=300)
    plt.show()
    plt.close()

    # Plot other losses (other losses - details.png)
    plt.figure(figsize=(10, 5))
    for metric, losses in metric_losses.items():
        plot_losses(losses, metric)
    plot_losses(grad_norms, 'Gradient Norm', ylabel='Gradient Norm')
    plt.title(f'Other Losses, Gradient Norm Over Time on {controller} Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{controller}_details.png'), dpi=300)
    plt.show()
    plt.close()


def get_models(models):
    if len(models) == 1:
        return models[0]
    else:
        return ', '.join(models[:-1]) + f', and {models[-1]}'


# To run the script: `torchrun --nproc_per_node=1 train.py`
def main() -> None:
    parser = argparse.ArgumentParser(description='Distributed Training Setup')
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['stu'], 
        choices=[
            'stu', 
            'transformer', 
            'mamba', 
            'jamba'
        ], 
        help='Models to train'
    )
    args = parser.parse_args()
    
    # Defaults specific to the Princeton HPC cluster; modify to your own setup.
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))
    gpus_per_node = int(os.environ.get('SLURM_GPUS_ON_NODE', 0))
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    device, local_rank, world_size = setup(rank, world_size, gpus_per_node)
    set_seed(1337 + local_rank)

    main_process = local_rank == 0
    if main_process:
        print(
            "Lyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant."
        )

    # General training hyperparameters
    train_batch_size: int = (
        48 // world_size
    )  # scale batch size for distributed training
    val_batch_size: int = (
        48 // world_size
    )  # scale batch size for distributed training
    num_epochs: int = 1
    eval_period: int = 30
    patience: int = 5
    checkpoint_dir: str = 'checkpoints'

    # Optimizer hyperparameters
    weight_decay: float = 1e-1
    m_y_learning_rate: float = 5e-5
    m_y_weight_decay: float = 0
    
    # STU hyperparameters
    d_model: int = 24
    d_target: int = 18
    num_layers: int = 6
    dropout: float = 0.25
    input_len: int = 1000
    num_eigh: int = 24
    auto_reg_k_u: int = 3
    auto_reg_k_y: int = 2
    learnable_m_y: bool = True
    stu_lr: float = 7.5e-4


    # Transformer hyperparameters
    n_layer: int = 6
    n_head: int = 1
    n_embd: int = 37
    scale: int = 4
    d_out: int = 29
    max_len: int = 1_000
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dropout: float = 0.25
    transformer_lr: float = 7.5e-4


    # Mamba hyperparameters
    # TBW

    # Jamba hyperparameters
    # TBW

    if main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists('plots/'):
            os.makedirs('plots/')

    controller = 'HalfCheetah-v1'
    train_inputs = f'../data/{controller}/3000/train_inputs.npy'
    train_targets = f'../data/{controller}/3000/train_targets.npy'
    val_inputs = f'../data/{controller}/3000/val_inputs.npy'
    val_targets = f'../data/{controller}/3000/val_targets.npy'

    # Get dataloaders
    train_loader = physics_data.get_dataloader(
        inputs=train_inputs,
        targets=train_targets,
        batch_size=train_batch_size,
        device=device,
        distributed=world_size > 1,
        rank=rank,
        num_replicas=world_size,
        num_workers=num_workers,
    )
    val_loader = physics_data.get_dataloader(
        inputs=val_inputs,
        targets=val_targets,
        batch_size=val_batch_size,
        device=device,
        distributed=world_size > 1,
        rank=rank,
        num_replicas=world_size,
        num_workers=num_workers,
    )
    num_steps: int = len(train_loader) * num_epochs
    warmup_steps: int = num_steps // 10

    models = {}
    optimizers = {}
    experiments = {}
    loss_fn = (
        HalfCheetahLoss()
        if controller == 'HalfCheetah-v1'
        else Walker2DLoss()
        if controller == 'Walker2D-v1'
        else AntLoss()
    )

    # Define the models based on flags
    if 'stu' in args.models:
        configs = {
            'd_model': d_model,
            'd_target': d_target,
            'num_layers': num_layers,
            'dropout': dropout,
            'input_len': input_len,
            'num_eigh': num_eigh,
            'auto_reg_k_u': auto_reg_k_u,
            'auto_reg_k_y': auto_reg_k_y,
            'learnable_m_y': learnable_m_y,
            'stu_lr': stu_lr,
            'loss_fn': loss_fn
        }
        
        stu_configs = STUConfigs(**configs)
        stu_model = model.Architecture(stu_configs).to(device)

        if world_size > 1:
            stu_model = DDP(
                stu_model,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        models['stu'] = stu_model.module if world_size > 1 else stu_model

        optimizers['stu'] = optimizer.get_optimizer(
            stu_model,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            learning_rate=stu_lr,
            weight_decay=weight_decay,
            m_y_learning_rate=m_y_learning_rate,
            m_y_weight_decay=m_y_weight_decay,
        )

        experiments['stu'] = experiment(
            model=stu_model, 
            loss_fn=loss_fn, 
            optimizer=optimizers['stu'], 
            device=device
        )

    if 'transformer' in args.models:
        configs = {
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'scale': scale,
            'd_out': d_out,
            'max_len': max_len,
            'bias': bias,
            'dropout': dropout,
            'loss_fn': loss_fn
        }
        transformer_configs = TransformerConfigs(**configs)
        transformer_model = Transformer(transformer_configs).to(device)

        if world_size > 1:
            transformer_model = DDP(transformer_model, device_ids=[local_rank], output_device=local_rank)

        models['transformer'] = transformer_model.module if world_size > 1 else transformer_model
        
        # TODO: Write a get_optimizer for Transformer in optimizer.py
        optimizers['transformer'] = torch.optim.AdamW(transformer_model.parameters(), lr=transformer_lr)

        experiments['transformer'] = experiment(
            model=transformer_model, 
            loss_fn=loss_fn, 
            optimizer=optimizers['transformer'], 
            device=device
        )
    
    # TO BE ADDED!
    # if 'mamba' in args.models:
    #     mamba_configs = MambaConfig(...)
    #     mamba_model = Mamba(mamba_configs)
    #     mamba_model = mamba_model.to(device)
    #     if world_size > 1:
    #         mamba_model = DDP(mamba_model, device_ids=[local_rank], output_device=local_rank)
    #     models['mamba'] = mamba_model.module if world_size > 1 else mamba_model
    #     optimizers['mamba'] = torch.optim.AdamW(mamba_model.parameters(), lr=args.mamba_lr)
    #     experiments['mamba'] = experiment(
    #         model=mamba_model, 
    #         loss_fn=loss_fn, 
    #         optimizer=optimizers['mamba'], 
    #         device=device
    #     )
    
    # if 'jamba' in args.models:
    #     jamba_configs = JambaConfig(...)
    #     jamba_model = Jamba(jamba_configs)
    #     jamba_model = jamba_model.to(device)
    #     if world_size > 1:
    #         jamba_model = DDP(jamba_model, device_ids=[local_rank], output_device=local_rank)
    #     models['jamba'] = jamba_model.module if world_size > 1 else jamba_model
    #     optimizers['jamba'] = torch.optim.AdamW(jamba_model.parameters(), lr=args.jamba_lr)
    #     experiments['jamba'] = experiment(
    #         model=jamba_model, 
    #         loss_fn=loss_fn, 
    #         optimizer=optimizers['jamba'], 
    #         device=device
    #     )
    
    best_val_losses = {model_name: float('inf') for model_name in args.models}
    patient_counters = {model_name: 0 for model_name in args.models}
    best_model_step = {model_name: 0 for model_name in args.models}
    best_checkpoints = {}

    # Initialize lists to store losses and metrics for each model
    train_losses = {model_name: [] for model_name in args.models}
    val_losses = {model_name: [] for model_name in args.models}
    grad_norms = {model_name: [] for model_name in args.models}
    metric_losses = {
        model_name: {
            'coordinate_loss': [],
            'orientation_loss': [],
            'angle_loss': [],
            'coordinate_velocity_loss': [],
            'angular_velocity_loss': [],
        }
        for model_name in args.models
    }

    if main_process:
        models = get_models(args.models)
        grmr = 'models' if len(args.models) > 1 else 'model'
        msg = f"Lyla: We'll be training the {models} {grmr} on the {controller} task with"
        if world_size > 1:
            print(
                f'{msg} {device} on rank {rank + 1}/{world_size}, '
                f'utilizing {world_size} distributed processes.'
            )
        else:
            print(f'{msg} {device} today.')

    pbar = (
        tqdm(
            range(num_epochs * len(train_loader)), desc='Training', unit='step'
        )
        if main_process
        else range(num_epochs * len(train_loader))
    )

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(train_loader):
            for model_name, exp in experiments.items():
                train_metrics = exp.step(inputs, targets)

                # Append the losses and metrics for each model
                train_losses[model_name].append(train_metrics['loss'])
                grad_norms[model_name].append(train_metrics['grad_norm'])
                for metric in metric_losses[model_name]:
                    if metric in train_metrics:
                        metric_losses[model_name][metric].append(train_metrics[metric])

                if main_process:
                    postfix_dict = {
                        f'{model_name}_tr_loss': train_metrics['loss'],
                        f'{model_name}_val_loss': val_losses[model_name][-1] if len(val_losses[model_name]) > 0 else None,
                        f'{model_name}_grd_nrm': train_metrics['grad_norm'],
                    }
                    for metric in train_metrics:
                        if metric in metric_losses[model_name]:
                            postfix_dict[f'{model_name}_{metric}'] = train_metrics[metric]
                    pbar.set_postfix(postfix_dict)

            if main_process:
                pbar.update(1)

            total_steps = epoch * len(train_loader) + step

            if total_steps > 0 and total_steps % eval_period == 0:
                for model_name, exp in experiments.items():
                    val_metrics = exp.evaluate(val_loader)
                    val_losses[model_name].append(val_metrics['loss'])

                    if world_size > 1:
                        # Gather evaluation metrics from all processes
                        gathered_metrics = [None] * world_size
                        torch.distributed.all_gather_object(gathered_metrics, val_metrics)

                        if main_process:
                            # Aggregate metrics across all processes
                            total_loss = sum(metric['loss'] for metric in gathered_metrics) / world_size
                            print(
                                f'\nLyla: Evaluating the {model_name} model on step {total_steps}'
                                f' Average Loss: {total_loss:.4f}.'
                            )
                            val_metrics = {'loss': total_loss}
                    else:
                        if main_process:
                            print(
                                f'\nLyla: Evaluating the {model_name} model on step {total_steps}'
                                f' Loss: {val_metrics["loss"]:.2f}.'
                            )

                    if main_process:
                        val_loss = val_metrics['loss']
                        if val_loss < best_val_losses[model_name]:
                            best_val_losses[model_name] = val_loss
                            best_model_step[model_name] = total_steps
                            patient_counters[model_name] = 0
                            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                            checkpoint_filename = f'{model_name}-checkpoint-step{total_steps}-{timestamp}.pt'
                            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                            best_checkpoints[model_name] = checkpoint_filename

                            torch.save(models[model_name].state_dict(), checkpoint_path)
                            print(
                                f'Lyla: Wow! We have a new personal best for the {model_name} model at step {total_steps}.'
                                f' The validation loss improved to: {val_loss:.4f}!'
                                f' Checkpoint saved as {checkpoint_path}'
                            )
                        else:
                            patient_counters[model_name] += 1
                            print(
                                f'Lyla: No improvement in validation loss for the {model_name} model'
                                f' for {patient_counters[model_name]} eval periods.'
                                f' Current best loss: {best_val_losses[model_name]:.4f}.'
                            )

                        if patient_counters[model_name] >= patience:
                            print(
                                f'Lyla: We have reached the patience limit of {patience}'
                                f' for the {model_name} model. Stopping the training early'
                                f' at step {total_steps}...'
                            )
                            dist.barrier()
                            return

    pbar.close()

    if main_process:
        print('\nLyla: Training completed!')
        for model_name in args.models:
            best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoints[model_name])
            models[model_name].load_state_dict(torch.load(best_checkpoint_path))
            print(f"\nLyla: Here's the best model information for the {model_name} model:")
            print(f'    Best model at step {best_model_step[model_name]}')
            print(f'    Best model validation loss: {best_val_losses[model_name]:.4f}')
            print(f'    Best model checkpoint saved at: {best_checkpoint_path}')

            # Save the training details to a file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            training_details = f'training_details_{model_name}_{timestamp}.txt'
            with open(training_details, 'w') as f:
                f.write(f'Training completed for {model_name} on {controller} at: {datetime.now()}\n')
                f.write(f'Best model step: {best_model_step[model_name]}\n')
                f.write(f'Best model validation loss: {best_val_losses[model_name]:.4f}\n')
                f.write(f'Best model checkpoint saved at: {best_checkpoint_path}\n')
            print(
                f'Lyla: Congratulations on completing the training run for the {model_name} model!'
                f' Details are saved in {training_details}.'
            )

        print('Lyla: It was a pleasure assisting you. Until next time!')

    if main_process:
        for model_name in args.models:
            plot_metrics(
                train_losses[model_name],
                val_losses[model_name],
                metric_losses[model_name],
                grad_norms[model_name],
                f'plots/{model_name}/', # TODO: Change this to a more centralized location
                controller,
                eval_period,
            )

if __name__ == '__main__':
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
