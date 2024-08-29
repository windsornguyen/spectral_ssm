# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: predict.py
# =============================================================================#

"""Prediction loop for Mamba-2 sequence prediction."""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file

from models.mamba.model import Mamba2, Mamba2Configs
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from losses.loss_cartpole import CartpoleLoss
from utils.dataloader import get_dataloader
from utils.dist import setup
from utils.dist_utils import get_data_parallel_group


def main():
    # Process command line flags
    parser = argparse.ArgumentParser(
        description="Inference script for sequence prediction"
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
        default="mujoco-v3",
        choices=["mujoco-v1", "mujoco-v2", "mujoco-v3"],
        help="Task to run inference on.",
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=1,
        help="Number of time steps to shift the targets from the inputs by.",
    )
    parser.add_argument(
        "--della",
        type=bool,
        default=True,
        help="Evaluating on the Princeton Della cluster. Defaults to True.",
        # NOTE: You MUST run with `torchrun` for this to work in the general setting.
    )
    args = parser.parse_args()

    # Defaults specific to the Princeton HPC cluster; modify to your own setup.
    device, local_rank, rank, world_size, main_process = setup(args)

    # Load the trained model
    model_path = f"mamba_{args.controller}_{args.task}.pt"

    # Shared hyperparameters
    d_conv: int = 4
    conv_init = None
    expand: int = 2
    d_ssm = None
    ngroups: int = 1 # TODO: What should this be set to?
    A_init_range: tuple[int, int] = (1, 16)
    activation = "silu"
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple[float, float] = (0.0, float("inf"))

    # Fused kernel and sharding options
    chunk_size: int = 256
    use_mem_eff_path: bool = True
    process_group = get_data_parallel_group()
    sequence_parallel: bool = True
    dtype: torch.dtype = torch.float32

    # MoE
    moe: bool = True
    num_experts: int = 3
    num_experts_per_timestep: int = 2

    # TODO: Experiment-specific hyperparameters
    # Data loader hyperparameters
    bsz: int = 2
    n_layers: int = 4
    d_model: int = 32
    mlp_scale: int = 10
    embd_scale: int = 3
    bias: bool = False
    dropout: float = 0.0
    conv_bias: bool = True
    loss_fn = nn.MSELoss()

    if args.task != "mujoco-v3":
        if args.controller == "Ant-v1":
            loss_fn = AntLoss()
        elif args.controller == "HalfCheetah-v1":
            loss_fn = HalfCheetahLoss()
        elif args.controller == "Walker2D-v1":
            loss_fn = Walker2DLoss()
        else:
            loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.MSELoss()

    # Task-specific hyperparameters
    if args.task == "mujoco-v1":
        d_in: int = 24 if args.controller != "Ant-v1" else 37
        # headdim: int = (expand * d_model) // world_size
        d_state: int = 128
        headdim: int = 1
        d_out: int = 18 if args.controller != "Ant-v1" else 29
        sl: int = 512

    elif args.task == "mujoco-v2":
        d_in: int = 18 if args.controller != "Ant-v1" else 29
        d_state: int = 130 if args.controller != "Ant-v1" else 128
        headdim: int = 1 if args.controller == "HalfCheetah-v1" else 1
        d_out: int = 18 if args.controller != "Ant-v1" else 29
        sl: int = 512

    elif args.task == "mujoco-v3":
        RESNET_D_OUT: int = 512  # ResNet-18 output dim
        RESNET_FEATURE_SIZE: int = 1
        d_out: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        headdim: int = (expand * d_model) // world_size
        sl: int = 300

    configs = Mamba2Configs(
        bsz=bsz,
        n_layers=n_layers,
        d_in=d_in,
        d_model=d_model,
        d_out=d_out,
        mlp_scale=mlp_scale,
        embd_scale=embd_scale,
        dropout=dropout,
        d_state=d_state,
        d_conv=d_conv,
        conv_init=conv_init,
        expand=expand,
        headdim=headdim,
        d_ssm=d_ssm,
        ngroups=ngroups,
        A_init_range=A_init_range,
        activation=activation,
        D_has_hdim=D_has_hdim,
        rmsnorm=rmsnorm,
        norm_before_gate=norm_before_gate,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_init_floor=dt_init_floor,
        dt_limit=dt_limit,
        bias=bias,
        conv_bias=conv_bias,
        chunk_size=chunk_size,
        use_mem_eff_path=use_mem_eff_path,
        process_group=process_group,
        sequence_parallel=sequence_parallel,

        # MoE
        moe=moe,
        num_experts=num_experts,
        num_experts_per_timestep=num_experts_per_timestep,

        loss_fn=loss_fn,
        controls={"task": args.task, "controller": args.controller},
        device=device,
        dtype=dtype,
        world_size=world_size
    )

    # Initialize and load the model
    model = Mamba2(configs).to(device)
    # model = torch.compile(model)
    state_dict = load_file(model_path, device="cuda:0")
    # state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load the test data
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        test_data = {
            "inputs": f"{base_path}/val_inputs.npy",
            "targets": f"{base_path}/val_targets.npy"
        }
    elif args.task == "mujoco-v3":
        test_data = torch.load(
            f"data/{args.task}/{args.controller}/{args.controller}_ResNet-18_val.pt",
            map_location=device,
        )
    else:
        raise ValueError("Invalid task")

    # Get test data loader
    test_loader = get_dataloader(
        model="mamba-2",
        data=test_data,
        task=args.task,
        controller=args.controller,
        bsz=args.bsz,
        shift=args.shift,
        preprocess=True,
        shuffle=False,
        pin_memory=True,
        distributed=False,
        local_rank=0,
        world_size=1,
        device=device,
    )

    # Run inference
    num_preds = 500
    predicted_states = []
    ground_truths = []
    losses = []
    init = sl
    
    # # Get the first batch to determine the steps
    # first_batch = next(iter(test_loader))
    # steps = first_batch[0].shape[1] - init
    steps = 50

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= num_preds:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            if args.task in ["mujoco-v1", "mujoco-v2"]:
                (
                    pred_states, pred_truths, 
                    (avg_loss, trajectory_losses),
                ) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,
                    steps=steps,
                    rollout_steps=1,
                    truth=0,
                )
            elif args.task == "mujoco-v3":
                (
                    pred_states, pred_truths, 
                    (avg_loss, trajectory_losses),
                ) = model.predict_states(
                    inputs=inputs,
                    targets=targets,
                    init=init,
                    steps=steps,
                    rollout_steps=1,
                    truth=0,
                )

            predicted_states.append(pred_states)
            ground_truths.append(pred_truths)
            losses.append(trajectory_losses)

    predicted_states = torch.cat(predicted_states, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0)
    losses = torch.cat(losses, dim=0)

    # Print out predictions and check if they're all the same
    for i in range(num_preds):
        print(f"\n Prediction {i+1}:")
        print(predicted_states[i, :5, 0])  # Print first 5 time steps of first feature

    # Check if all predictions are the same
    all_same = True
    for i in range(1, num_preds):
        if not torch.allclose(predicted_states[0], predicted_states[i], atol=1e-6):
            all_same = False
            break

    print(f"\nAll predictions are the same: {all_same}")

    if all_same:
        print(
            "All predictions are identical. This might indicate an issue with the model or data processing."
        )
    else:
        print("Predictions differ, which is expected for different inputs.")

    print(f"Shape of predicted states: {predicted_states.shape}")
    print(f"Shape of ground truths: {ground_truths.shape}")
    print(f"Shape of losses: {losses.shape}")

    # Save predictions and ground truths
    np.save(
        f"mamba_{args.controller}_{args.task}_predictions.npy",
        predicted_states.cpu().numpy(),
    )
    np.save(
        f"mamba_{args.controller}_{args.task}_ground_truths.npy",
        ground_truths.cpu().numpy(),
    )
    np.save(f"mamba_{args.controller}_{args.task}_losses.npy", losses.cpu().numpy())
    print(
        f"Predictions, ground truths, and losses saved to 'mamba_{args.controller}_{args.task}_predictions.npy', 'mamba_{args.controller}_{args.task}_ground_truths.npy', and 'mamba_{args.controller}_{args.task}_losses.npy' respectively."
    )


if __name__ == "__main__":
    main()
