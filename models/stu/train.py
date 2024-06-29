# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: train.py
# =============================================================================#

"""Training loop for STU sequence prediction."""

import argparse
from datetime import datetime
import os

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from time import time

from torch.nn import MSELoss
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from utils.dataloader import get_dataloader, split_data
from utils import experiment as exp, optimizer as opt
from models.stu.model import SSSM, SSSMConfigs
from utils.colors import Colors, colored_print
from utils.dist import setup, cleanup


def save_results(task, ctrl, data, name, ts, directory="results", prefix="sssm", meta=None):
    """
    Save data to a file with enhanced flexibility and metadata support.

    Args:
        task (str): Task name.
        ctrl (str): Controller name.
        data (list or dict): Data to save.
        name (str): Data category name.
        ts (str): Timestamp.
        dir (str): Base directory for saving files.
        prefix (str): File name prefix.
        meta (dict): Additional metadata to save.

    Returns:
        str: Path of the saved file.
    """
    path = os.path.join(directory, task, prefix)
    os.makedirs(path, exist_ok=True)

    fname = f"{prefix}-{ctrl}-{name}-{ts}.txt"
    fpath = os.path.join(path, fname)

    with open(fpath, "w") as f:
        if meta:
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
            f.write("\n")

        if isinstance(data, dict):
            for k, v in data.items():
                f.write(f"# {k}\n")
                for item in v:
                    f.write(f"{item}\n")
                f.write("\n")
        else:
            for item in data:
                f.write(f"{item}\n")

    print(f"Data saved to {fpath}")
    return fpath


# Example: `torchrun -m --nproc_per_node=1 models.stu.train_stu --controller Ant-v1 --task mujoco-v3`
def main() -> None:
    torch.set_float32_matmul_precision("high")  # Enable CUDA TensorFloat-32

    # Process command line flags
    parser = argparse.ArgumentParser(
        description="Training script for sequence prediction"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="Ant-v1",
        choices=["Ant-v1", "HalfCheetah-v1", "Walker2D-v1"],
        help="Controller to use for the MuJoCo environment. Defaults to Ant-v1.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mujoco-v3",
        choices=[
            "mujoco-v1",  # Predict state trajectories, incl. controls as input
            "mujoco-v2",  # Predict state trajectories, w/o incl. controls as input
            "mujoco-v3",  # Predict state trajectories using a unified representation
        ],
        help="Task to train on. Defaults to mujoco-v3.",
    )
    parser.add_argument(
        "--della",
        type=bool,
        default=True,
        help="Training on the Princeton Della cluster. Defaults to True.",
        # NOTE: You MUST run with `torchrun` for this to work in the general setting.
    )
    args = parser.parse_args()

    controller = args.controller
    task = {
        "mujoco-v1": args.task == "mujoco-v1",
        "mujoco-v2": args.task == "mujoco-v2",
        "mujoco-v3": args.task == "mujoco-v3",
    }

    # Defaults specific to the Princeton HPC cluster; modify to your own setup.
    # device, local_rank, rank, world_size, num_workers, main_process = setup(args)
    device, local_rank, rank, world_size, main_process = setup(args)

    if main_process:
        colored_print(
            "\nLyla: Greetings! I'm Lyla, your friendly neighborhood AI training assistant.",
            Colors.OKBLUE,
        )

    # Prepare directories for training and plotting
    checkpoint_dir: str = "checkpoints"
    if main_process:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists("results/"):
            os.makedirs("results/")

    # Shared hyperparameters
    # TODO: Make these argparse arguments eventually else default to these.
    n_layers: int = 4
    scale: int = 4
    bias: bool = False
    dropout: float = 0.10
    num_eigh: int = 24
    k_u: int = 3
    k_y: int = 2
    learnable_m_y: bool = True

    if not task["mujoco-v3"]:
        if controller == "Ant-v1":
            loss_fn = AntLoss()
        elif controller == "HalfCheetah-v1":
            loss_fn = HalfCheetahLoss()
        elif controller == "Walker2D-v1":
            loss_fn = Walker2DLoss()
        else:
            loss_fn = None
    else:
        loss_fn = MSELoss()

    # Task-specific hyperparameters
    if task["mujoco-v1"]:
        n_embd: int = 24 if controller != "Ant-v1" else 37
        d_in = n_embd # TODO: d_in is not exactly the same as n_embd
        d_out: int = 18 if controller != "Ant-v1" else 29
        sl: int = 1_000

        configs = SSSMConfigs(
            n_layers=n_layers,
            n_embd=n_embd,
            d_in=d_in,
            d_out=d_out,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            num_eigh=num_eigh,
            k_u=k_u,
            k_y=k_y,
            learnable_m_y=learnable_m_y,
            loss_fn=loss_fn,
            controls={"task": "mujoco-v1", "controller": controller},
        )

    elif task["mujoco-v2"]:
        n_embd: int = 18 if controller != "Ant-v1" else 29
        d_in = n_embd # TODO: d_in is not exactly the same as n_embd
        d_out = n_embd
        sl: int = 1_000
        configs = SSSMConfigs(
            n_layers=n_layers,
            n_embd=n_embd,
            d_in=d_in,
            d_out=d_out,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            num_eigh=num_eigh,
            k_u=k_u,
            k_y=k_y,
            learnable_m_y=learnable_m_y,
            loss_fn=loss_fn,
            controls={"task": "mujoco-v2", "controller": controller},
        )

    elif task["mujoco-v3"]:
        RESNET_D_OUT: int = 512  # ResNet-18 output dim
        RESNET_FEATURE_SIZE: int = 1
        d_out: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        n_embd = d_out
        d_in = n_embd # TODO: d_in is not exactly the same as n_embd
        sl: int = 300

        configs = SSSMConfigs(
            n_layers=n_layers,
            n_embd=n_embd,
            d_in=d_in,
            d_out=d_out,
            sl=sl,
            scale=scale,
            bias=bias,
            dropout=dropout,
            num_eigh=num_eigh,
            k_u=k_u,
            k_y=k_y,
            learnable_m_y=learnable_m_y,
            loss_fn=loss_fn,
            controls={"task": "mujoco-v3", "controller": controller},
        )

    model = SSSM(configs).to(device)
    # model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    stu_model = model.module if world_size > 1 else model

    # Data loader hyperparameters
    bsz: int = 80
    preprocess: bool = True

    # TODO: Put in v2 data (no controls)
    mujoco_v1_base = f"data/mujoco-v1/{args.controller}/"
    mujoco_v2_base = f"data/mujoco-v2/{args.controller}/"
    mujoco_v3_base = f"data/mujoco-v3/{args.controller}/"

    # Initialize dataset variable
    dataset = None

    # Handle mujoco-v1 and mujoco-v2 tasks  
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = mujoco_v1_base if args.task == "mujoco-v1" else mujoco_v2_base
        train_data = {
            "inputs": f"{base_path}/train_inputs.npy",
            "targets": f"{base_path}/train_targets.npy",
        }
        val_data = {
            "inputs": f"{base_path}/val_inputs.npy",
            "targets": f"{base_path}/val_targets.npy",
        }
    elif args.task == "mujoco-v3":
        dataset = torch.load(
            f"{mujoco_v3_base}{args.controller}_ResNet-18.pt", map_location=device
        )
        train_data, val_data = split_data(dataset)
    else:
        raise ValueError("Invalid task")

    # TODO: May need to condition the dataloader shift on mujoco-v3 task only?
    shift = 1
    train_loader = get_dataloader(
        data=train_data,
        task=args.task,
        bsz=bsz,
        shift=shift,
        preprocess=preprocess,
        shuffle=True,
        pin_memory=True,
        distributed=world_size > 1,
        rank=local_rank,
        world_size=world_size,
        device=device,
    )

    val_loader = get_dataloader(
        data=val_data,
        task=args.task,
        bsz=bsz,
        shift=shift,
        preprocess=preprocess,
        shuffle=False,
        pin_memory=True,
        distributed=world_size > 1,
        rank=local_rank,
        world_size=world_size,
        device=device,
    )

    # General training hyperparameters
    training_stu = True
    num_epochs: int = 3
    steps_per_epoch = len(train_loader)
    num_steps: int = steps_per_epoch * num_epochs
    dilation: int = 1
    warmup_steps: int = num_steps // 8
    eval_period: int = num_steps // 16

    if main_process:
        colored_print(f"\nUsing batch size: {bsz}", Colors.OKCYAN)
        colored_print(f"Number of epochs: {num_epochs}", Colors.OKCYAN)
        colored_print(f"Steps per epoch: {steps_per_epoch}", Colors.OKCYAN)
        colored_print(f"=> Number of training steps: {num_steps}", Colors.OKCYAN)

    # General training variables
    patient_counter = 0
    best_val_loss = float("inf")
    best_model_step = 0
    best_checkpoint = None

    # Number of non-improving eval periods before early stopping
    patience: int = 10

    # Optimizer hyperparameters
    weight_decay: float = 1e-1
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1
    betas = (0.9, 0.95)
    eps = 1e-8
    use_amsgrad = False
    optimizer_settings = (
        warmup_steps,
        num_steps,
        max_lr,
        min_lr,
        betas,
        eps,
        weight_decay,
        use_amsgrad,
    )

    training_run = exp.Experiment(
        model=stu_model,
        task=task,
        loss_fn=loss_fn,
        bsz=bsz,
        sl=sl,
        optimizer_settings=optimizer_settings,
        training_stu=training_stu,
        world_size=world_size,
        main_process=main_process,
        device=device,
    )

    # Lists to store losses and metrics
    train_losses = []
    val_losses = []
    val_time_steps = []
    grad_norms = []

    if not task["mujoco-v3"]:
        metric_losses = {
            "coordinate_loss": [],
            "orientation_loss": [],
            "angle_loss": [],
            "coordinate_velocity_loss": [],
            "angular_velocity_loss": [],
        }

    if main_process:
        msg = f"\nLyla: We'll be training the SSSM model on the {args.task} task with {controller}."
        if world_size > 1:
            colored_print(
                f"{msg} {device} on rank {rank + 1}/{world_size}"
                f" utilizing {world_size} distributed processes.",
                Colors.HEADER,
            )
        else:
            colored_print(f"{msg} {device} today.", Colors.OKCYAN)

    # Training loop
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(train_loader):
            relative_step = epoch * steps_per_epoch + step
            last_step = relative_step == num_steps - 1

            # Perform a training step
            train_results = training_run.step(inputs, targets, relative_step)
            train_losses.append(train_results["loss"])
            grad_norms.append(train_results["grad_norm"])

            if not task["mujoco-v3"]:
                for k, v in train_results.items():
                    if k in metric_losses:
                        metric_losses[k].append(v)

            # Periodically evaluate the model on validation set
            if relative_step % (eval_period // dilation) == 0 or last_step:
                if main_process:
                    colored_print(
                        f"\nLyla: Evaluating the SSSM model on step {relative_step}.",
                        Colors.OKCYAN,
                    )
                val_metrics = training_run.evaluate(val_loader)

                val_losses.append(val_metrics["loss"])
                val_time_steps.append(relative_step)

                if main_process:
                    colored_print(
                        f"\nValidation Loss: {val_metrics['loss']:.4f}.",
                        Colors.OKCYAN,
                    )

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        best_model_step = relative_step
                        patient_counter = 0

                        # Construct paths for model checkpoint and extra info
                        model_checkpoint = f"sssm-{controller}-model_step-{relative_step}-{timestamp}-48l.pt"
                        model_path = os.path.join(checkpoint_dir, model_checkpoint)

                        extra_info = f"sssm-{controller}-other_step-{relative_step}-{timestamp}-48l.pt"
                        extra_info_path = os.path.join(checkpoint_dir, extra_info)

                        best_checkpoint = (model_checkpoint, extra_info)

                        # Save model state dict using safetensors
                        save_file(training_run.model.state_dict(), model_path)

                        # Save optimizer state and other metadata using torch.save
                        torch.save(
                            {
                                "optimizer": training_run.optimizer.state_dict(),
                                "configs": training_run.model.configs,
                                "step": relative_step,
                                "val_loss": val_metrics["loss"],
                                "metrics": val_metrics,
                                "timestamp": timestamp,
                                "rng_state_pytorch": torch.get_rng_state(),
                                "rng_state_numpy": np.random.get_state(),
                                "rng_state_cuda": torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available()
                                else None,
                            },
                            extra_info_path,
                        )

                        colored_print(
                            f"Lyla: Wow! We have a new personal best for the SSSM model at step {relative_step}. "
                            f"The validation loss improved to: {val_metrics['loss']:.4f}! "
                            f"Model checkpoint saved as {model_path} and other data saved as {extra_info_path}",
                            Colors.OKGREEN,
                        )
                    else:
                        patient_counter += 1
                        colored_print(
                            f"Lyla: No improvement in validation loss for the SSSM model for {patient_counter} eval periods. Current best loss: {best_val_loss:.4f}.",
                            Colors.WARNING,
                        )

                if patient_counter >= patience:
                    if main_process:
                        colored_print(
                            f"Lyla: We have reached the patience limit of {patience} for the SSSM model. Stopping the training early at step {relative_step}...",
                            Colors.FAIL,
                        )
                    break

            # Logging
            if main_process and relative_step % 5 == 0:
                colored_print(f"\nStep {relative_step:5d}", Colors.HEADER)
                colored_print(
                    f"Train Loss: {train_results['loss']:.6f} | Gradient Norm: {train_results['grad_norm']:.4f} | "
                    f"Step Time: {train_results['step_time']*1000:.4f}ms | Tokens/sec: {train_results['tokens_per_sec']:.4f}",
                    Colors.OKBLUE,
                )

                # Report learning rates for each parameter group
                # TODO: Check that the learning rates are being adjusted correctly.
                lr_reports = []
                for param_group in training_run.optimizer.param_groups:
                    lr_reports.append(f"{param_group['name']}: {param_group['lr']:.6f}")
                lr_report = "Learning Rates: " + " | ".join(lr_reports)
                colored_print(lr_report, Colors.OKCYAN)

        if patient_counter >= patience:
            break

    # Post-training processing
    if main_process:
        if best_checkpoint:
            model_checkpoint, extra_info = best_checkpoint
            best_model_path = os.path.join(checkpoint_dir, model_checkpoint)
            best_model_extra_info_path = os.path.join(checkpoint_dir, extra_info)

            if dist.is_initialized():
                # Load the best checkpoint on the main process and broadcast it to all processes
                if main_process:
                    # Load model state dict
                    state_dict = load_file(best_model_path, device=rank)
                    training_run.model.load_state_dict(state_dict)

                    # Load optimizer and other data
                    other_data = torch.load(
                        best_model_extra_info_path, map_location=f"cuda:{rank}"
                    )
                    training_run.optimizer.load_state_dict(other_data["optimizer"])
                dist.barrier()
            else:
                # Load model state dict
                state_dict = load_file(best_model_path, device="cpu")
                training_run.model.load_state_dict(state_dict)

                # Load optimizer and other data
                other_data = torch.load(best_model_extra_info_path, map_location="cpu")
                training_run.optimizer.load_state_dict(other_data["optimizer"])

            print("\nLyla: Here's the best model information for the SSSM model:")
            print(f"    Best model at step {best_model_step}")
            print(f"    Best model validation loss: {best_val_loss:.4f}")
            print(f"    Best model checkpoint saved at: {best_model_path}")
            print(f"    Best other data saved at: {best_model_extra_info_path}")

            # Save the training details to a file
            training_details = f"training_details_sssm_{timestamp}.txt"
            with open(training_details, "w") as f:
                f.write(
                    f"Training completed for SSSM on {args.task} with {controller} at: {datetime.now()}\n"
                )
                f.write(f"Best model step: {best_model_step}\n")
                f.write(f"Best model validation loss: {best_val_loss:.4f}\n")
                f.write(f"Best model checkpoint saved at: {best_model_path}\n")
                f.write(
                    f"Best model's extra info data saved at: {best_model_extra_info_path}\n"
                )
            print(
                f"Lyla: Congratulations on completing the training run for the SSSM model! Details are saved in {training_details}."
            )
        else:
            colored_print(
                "\nLyla: No best checkpoint found for the SSSM model. The model did not improve during training.",
                Colors.WARNING,
            )

        # Save the final results
        if main_process:
            save_results(args.task, controller, train_losses, "train_losses", timestamp)
            save_results(
                args.task,
                controller,
                val_losses,
                "val_losses",
                timestamp,
            )
            save_results(args.task, controller, val_time_steps, "val_time_steps", timestamp)
            save_results(args.task, controller, grad_norms, "grad_norms", timestamp)

            if not task["mujoco-v3"]:
                for metric, losses in metric_losses.items():
                    save_results(args.task, controller, losses, metric, timestamp)

            colored_print(
                "Lyla: It was a pleasure assisting you. Until next time!",
                Colors.OKGREEN,
            )

if __name__ == "__main__":
    main()
    if dist.is_initialized():
        cleanup()
