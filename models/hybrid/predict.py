# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: predict.py
# =============================================================================#

"""Prediction loop for Hybrid sequence prediction."""

import argparse
import torch
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file

from models.hybrid.model import SpectralHybrid, SpectralHybridConfigs

from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss

from utils.dataloader import get_dataloader


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
    # dilated attention
    parser.add_argument(
        "--dilated_attn",
        type=bool,
        default=False,
        help="Whether to use dilated attention. Defaults to False.",
    )
    parser.add_argument(
        "--segment_lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Segment lengths for dilated attention. Defaults to [128, 256, 512].",
    )
    parser.add_argument(
        "--dilated_ratios",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Dilation ratios for dilated attention. Defaults to [1, 2, 4].",
    )
    parser.add_argument(
        "--seq_parallel",
        type=bool,
        default=True,
        help="Whether to use sequence parallelism. Defaults to True.",
    )
    parser.add_argument(
        "--xpos_rel_pos",
        type=bool,
        default=True,
        help="Whether to use relative positional embeddings. Defaults to True.",
    )
    parser.add_argument(
        "--xpos_scale_base",
        type=int,
        default=512,
        help="Scale base for positional embeddings. Defaults to 512.",
    )
    parser.add_argument(
        "--rms_norm_eps",
        type=float,
        default=1e-5,
        help="Epsilon for root mean square normalization. Defaults to 1e-5.",
    )
    parser.add_argument(
        "--multiway",
        type=bool,
        default=False,
        help="Whether to use multiway attention. Defaults to False.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = f"hybrid_{args.controller}_{args.task}.pt"

    # Shared hyperparameters
    # STU settings
    num_eigh: int = 16
    k_y: int = 2
    k_u: int = 3
    alpha: float = 0.9  # 0.9 deemed "uniformly optimal" in the paper
    use_ar_y: bool = False
    use_ar_u: bool = False
    use_hankel_L: bool = False
    use_flash_fft: bool = False
    use_approx: bool = False

    # Transformer settings
    # pct_attn: float = 0.5 # Percentage of layers using attention
    flash_attn: bool = True # Whether to use FlashAttention-2 or not

    # MoE
    moe: bool = True
    num_experts: int = 3
    num_experts_per_timestep: int = 2

    # General training settings
    n_layers: int = 2
    d_model: int = 32
    mlp_scale: int = 8
    embd_scale: int = 3
    bias: bool = False
    dropout: float = 0.0 # Convert all these into argparses eventually
    flash_attn: bool = True
    use_sq_relu: bool = False # Performs BETTER with Squared ReGLU\
    use_alibi: bool = False

    if args.task != "mujoco-v3":
        if args.controller == "Ant-v1":
            loss_fn = AntLoss()
        elif args.controller == "HalfCheetah-v1":
            loss_fn = HalfCheetahLoss()
        elif args.controller == "Walker2D-v1":
            loss_fn = Walker2DLoss()
        else:
            loss_fn = None
    else:
        loss_fn = MSELoss()

    # Task-specific hyperparameters
    if args.task == "mujoco-v1":
        d_in: int = 24 if args.controller != "Ant-v1" else 37
        n_heads: int = 8 if args.controller != "Ant-v1" else 1
        d_out: int = 18 if args.controller != "Ant-v1" else 29
        sl: int = 512   # sl supported by flashfftconv

    elif args.task == "mujoco-v2":
        d_in: int = 18 if args.controller != "Ant-v1" else 29
        n_heads: int = 9 if args.controller != "Ant-v1" else 1
        d_out = d_in
        sl: int = 512

    elif args.task == "mujoco-v3":
        RESNET_D_OUT: int = 512  # ResNet-18 output dim
        RESNET_FEATURE_SIZE: int = 1
        d_out: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        d_in: int = RESNET_D_OUT * RESNET_FEATURE_SIZE**2
        n_heads: int = 16
        sl: int = 300

    window_size: int = sl # Global attention
    configs = SpectralHybridConfigs(
        # STU settings
        d_in=d_in,
        d_out=d_out,
        mlp_scale=mlp_scale,
        embd_scale=embd_scale,
        num_eigh=num_eigh,
        k_y=k_y,
        k_u=k_u,
        alpha=alpha,
        use_ar_y=use_ar_y,
        use_ar_u=use_ar_u,
        use_hankel_L=use_hankel_L,
        use_flash_fft=use_flash_fft,
        use_approx=use_approx,

        # Transformer settings
        d_model=d_model,
        n_heads=n_heads,
        flash_attn=flash_attn,
        use_sq_relu=use_sq_relu,
        window_size=window_size,
        use_alibi=use_alibi,
        
        # MoE
        moe=moe,
        num_experts=num_experts,
        num_experts_per_timestep=num_experts_per_timestep,
        
        # General training settings
        sl=sl,
        n_layers=n_layers,
        
        bias=bias,
        dropout=dropout,
        loss_fn=loss_fn,
        controls={"task": args.task, "controller": args.controller},
        device=device,
    )

    # Initialize and load the model
    model = SpectralHybrid(configs).to(device)
    # model = torch.compile(model)
    # state_dict = load_file(model_path, device="cuda:0")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load the test data
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        test_data = {
            "inputs": f"{base_path}/val_inputs.npy",
            "targets": f"{base_path}/val_targets.npy",
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
        model="hybrid",
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
        f"hybrid_{args.controller}_{args.task}_predictions.npy",
        predicted_states.cpu().numpy(),
    )
    np.save(
        f"hybrid_{args.controller}_{args.task}_ground_truths.npy",
        ground_truths.cpu().numpy(),
    )
    np.save(f"hybrid_{args.controller}_{args.task}_losses.npy", losses.cpu().numpy())
    print(
        f"Predictions, ground truths, and losses saved to 'hybrid_{args.controller}_{args.task}_predictions.npy', 'hybrid_{args.controller}_{args.task}_ground_truths.npy', and 'hybrid_{args.controller}_{args.task}_losses.npy' respectively."
    )


if __name__ == "__main__":
    main()
