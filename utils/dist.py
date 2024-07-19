# =============================================================================#
# Authors: Windsor Nguyen
# File: dist.py
# =============================================================================#

"""Helper functions for distributed training."""

import os
import torch
import torch.distributed as dist
import random
import numpy as np

from utils.colors import Colors, colored_print


def set_seed(seed: int, main_process: bool) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    if main_process:
        colored_print(f"Random seed set to {seed}", Colors.OKCYAN)


def setup_della(
    rank: int, world_size: int, gpus_per_node: int
) -> tuple[torch.device, int, int, bool]:
    """
    Set up distributed environment for Princeton clusters.

    Returns:
        tuple[
            torch.device,
            int,
            int,
            bool
        ]: Device, local rank, world size, and main process flag
    """
    local_rank = rank % gpus_per_node if gpus_per_node > 0 else 0
    device = torch.device("cpu")
    backend = "gloo"

    if world_size > 1 and "SLURM_PROCID" in os.environ:
        if torch.cuda.is_available() and gpus_per_node > 0:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if rank == 0:
            colored_print(
                f"Initialized Princeton distributed setup with {world_size} processes",
                Colors.OKGREEN,
            )
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if rank == 0:
            colored_print("\nUsing CUDA device for Princeton setup.", Colors.OKBLUE)
    elif rank == 0:
        colored_print("Using CPU for Princeton setup", Colors.WARNING)

    main_process = rank == 0
    return device, local_rank, world_size, main_process


def setup_general() -> tuple[torch.device, int, int, bool]:
    """
    Set up distributed environment for general use.

    Returns:
        tuple[
            torch.device,
            int,
            int,
            bool
        ]: Device, local rank, world size, and main process flag
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(local_rank)
            if rank == 0:
                colored_print(
                    f"Initialized general distributed setup with {world_size} processes",
                    Colors.OKGREEN,
                )
        else:
            local_rank = rank = 0
            world_size = 1
            colored_print("Using single CUDA device for general setup", Colors.OKBLUE)
    else:
        device = torch.device("cpu")
        local_rank = rank = 0
        world_size = 1
        colored_print("Using CPU for general setup", Colors.WARNING)

    main_process = rank == 0
    return device, local_rank, world_size, main_process


def setup(args) -> tuple[torch.device, int, int, int, bool]:
    """
    Initialize distributed training environment based on args.

    Args:
        args: Command-line arguments

    Returns:
        tuple[
            torch.device,
            int,
            int,
            int,
            bool
        ]: Device, local rank, world size, number of workers, and main process flag
    """
    if args.della:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("SLURM_PROCID", 0))
        gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
        device, local_rank, world_size, main_process = setup_della(
            rank, world_size, gpus_per_node
        )
    else:
        device, local_rank, world_size, main_process = setup_general()
        num_workers = 4  # Default for non-Princeton setups, adjust as needed
        rank = 0 if main_process else None

    if main_process:
        colored_print(
            f"Distributed {'Della' if args.della else 'general'} setup initiated.",
            Colors.HEADER,
        )
        colored_print(
            f"Initialization complete. Device: {device}, World Size: {world_size}, Workers: {num_workers}.",
            Colors.BOLD,
        )

    set_seed(1337 + local_rank, main_process)
    # return device, local_rank, rank, world_size, num_workers, main_process
    return device, local_rank, rank, world_size, main_process


def cleanup() -> None:
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        if dist.get_rank() == 0:
            colored_print("Distributed process group destroyed.", Colors.OKBLUE)
