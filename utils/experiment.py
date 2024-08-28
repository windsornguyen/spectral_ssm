# ==============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: experiment.py
# ==============================================================================#

"""Utilities for running an experiment for Spectral SSM."""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.optim import AdamW
from utils.colors import Colors, colored_print

# Loss landscape visualization
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from vtk import vtkStructuredGrid, vtkPoints, vtkDoubleArray, vtkXMLStructuredGridWriter
from scipy import interpolate


class Experiment:
    """
    Initializes and maintains the experiment state.
    """

    def __init__(
        self,
        model: nn.Module,
        task: dict[str, bool],
        loss_fn: nn.Module,
        bsz: int,
        sl: int,
        optimizer_settings: tuple[int, int, float, float],
        training_stu: bool = False,
        world_size: int = 1,
        main_process: bool = False,
        device: torch.device = None,
    ) -> None:
        """
        Initialize an experiment.

        Args:
            model (nn.Module): A PyTorch model.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            device (torch.device): The device to run the model on.
        """
        self.model = model
        self.device = device
        self.task = task
        self.loss_fn = loss_fn
        (
            self.warmup_steps,
            self.num_steps,
            self.max_lr,
            self.min_lr,
            self.betas,
            self.eps,
            self.weight_decay,
            self.use_amsgrad,
        ) = optimizer_settings

        # Additional information to process
        self.bsz = bsz
        self.sl = sl
        self.main_process = main_process
        self.world_size = world_size

        # If training STU
        if training_stu:
            self.m_y_learning_rate = 5e-5
            self.m_y_weight_decay = 0

        self.optimizer = self.get_optimizer(
            self.max_lr, self.betas, self.eps, self.weight_decay, self.use_amsgrad
        )

        self.model.to(self.device)

    def get_optimizer(self, lr, betas, eps, weight_decay, use_amsgrad):
        param_groups = []
        m_y_params = []
        # stu_params = {f"stu_{i}": [] for i in range(1, 5)}
        # stu_mlp_params = {}
        default_params = []

        # # Define different learning rates for each STU
        # stu_lr_multipliers = {
        #     "stu_1": 1.0,
        #     "stu_2": 0.7,
        #     "stu_3": 0.4,
        #     "stu_4": 0.1,
        # }

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith("m_y"):
                    m_y_params.append(param)
                # elif any(f"stu_{i}" in name for i in range(1, 5)):
                #     stu_number = next(i for i in range(1, 5) if f"stu_{i}" in name)
                #     stu_params[f"stu_{stu_number}"].append(param)
                # elif "stu_mlp_pairs" in name:
                #     pair_index = int(name.split('.')[2])  
                #     if pair_index not in stu_mlp_params:
                #         stu_mlp_params[pair_index] = []
                #     stu_mlp_params[pair_index].append(param)
                else:
                    default_params.append(param)

        # # Add parameter groups for STUs with their specific learning rates
        # for stu_name, params in stu_params.items():
        #     if params:
        #         stu_lr = lr * stu_lr_multipliers[stu_name]
        #         param_groups.append(
        #             {
        #                 "name": stu_name,
        #                 "params": params,
        #                 "lr": stu_lr,
        #                 "weight_decay": weight_decay,
        #             }
        #         )

        # # Add parameter groups for STU-MLP pairs with decreasing learning rates
        # stu_lr_multipliers = [1.0, 1.0, 1.0, 1.0]  # Adjust as needed
        # for pair_index, params in stu_mlp_params.items():
        #     multiplier = stu_lr_multipliers[pair_index] if pair_index < len(stu_lr_multipliers) else stu_lr_multipliers[-1]
        #     param_groups.append({
        #         "name": f"stu_mlp_pair_{pair_index}",
        #         "params": params,
        #         "lr": lr * multiplier,
        #         "weight_decay": weight_decay
        #     })

        # Add parameter groups for m_y and default params
        if m_y_params:
            param_groups.extend(
                [
                    {
                        "name": "default",
                        "params": default_params,
                        "lr": self.max_lr,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "name": "m_y",
                        "params": m_y_params,
                        "lr": self.m_y_learning_rate,
                        "weight_decay": self.m_y_weight_decay,
                    },
                ]
            )

        decay_params = [p for p in default_params if p.dim() >= 2]
        nodecay_params = [p for p in default_params if p.dim() < 2]
        param_groups.extend(
            [
                {
                    "name": "decay",
                    "params": decay_params,
                    "lr": self.max_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "name": "no_decay",
                    "params": nodecay_params,
                    "lr": self.max_lr,
                    "weight_decay": 0.0,
                },
            ]
        )

        if self.main_process:
            for group in param_groups:
                colored_print(
                    f'\nOptimizer | Group {group["name"]}: '
                    f'{len(group["params"])} tensors, '
                    f'{sum(p.numel() for p in group["params"]):,} parameters, '
                    f'lr: {group["lr"]}, weight_decay: {group["weight_decay"]}',
                    Colors.HEADER,
                )

            lr_reports = [f"{group['name']}: {group['lr']:.6f}" for group in param_groups]
            lr_report = "Learning Rates: " + " | ".join(lr_reports)
            colored_print(lr_report, Colors.OKCYAN)

        fused_available = "fused" in inspect.signature(AdamW).parameters
        use_fused = fused_available and self.device.type == "cuda"

        if self.main_process:
            colored_print(f"Optimizer | Using fused AdamW?: {use_fused}", Colors.HEADER)

        return AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=use_amsgrad,
            fused=use_fused,
        )


    def get_lr(
        self,
        it,
        warmup_steps,
        num_steps,
        max_lr,
        min_lr,
    ):
        """
        Custom learning rate scheduler: linear warmup and cosine decay.
        """
        # 1. Linear warmup for warmup_steps steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps

        # 2. If it > lr_decay_iters, return min learning rate
        if it > num_steps:
            return min_lr

        # 3. If in between, cosine decay to down to min learning rate
        decay_ratio = (it - warmup_steps) / (num_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    def step(
        self, inputs: torch.Tensor, targets: torch.Tensor, relative_step: int
    ) -> dict[str, float]:
        """
        Perform a single training step.

        Args:
            inputs (torch.Tensor): A batch of input data.
            targets (torch.Tensor): A batch of target labels.
            relative_step (int): The current step relative to the start of training.

        Returns:
            dict[str, float]: A dictionary of metrics for the training step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        t0 = time()

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            preds, loss_info = self.model(inputs, targets)

        if isinstance(loss_info, tuple):
            loss, *step_metrics = loss_info
        else:
            loss = loss_info

        loss.backward()

        if self.world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        # Clip global norm of gradient at 1.0, per the GPT-3 paper
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Update learning rates for each parameter group
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "m_y":
                param_group["lr"] = self.get_lr(
                    relative_step,
                    self.warmup_steps,
                    self.num_steps,
                    self.m_y_learning_rate,
                    self.min_lr,
                )
            # elif param_group["name"].startswith("stu_"):
            #     stu_base_lr = self.get_lr(
            #         relative_step,
            #         self.warmup_steps,
            #         self.num_steps,
            #         self.max_lr,
            #         self.min_lr,
            #     )
            #     stu_multiplier = {
            #         "stu_1": 1.0,
            #         "stu_2": 0.7,
            #         "stu_3": 0.4,
            #         "stu_4": 0.1,
            #     }[param_group["name"]]
            #     param_group["lr"] = stu_base_lr * stu_multiplier
            # elif param_group["name"].startswith("stu_mlp_pair_"):
            #     pair_index = int(param_group["name"].split('_')[-1])
            #     stu_base_lr = self.get_lr(
            #         relative_step,
            #         self.warmup_steps,
            #         self.num_steps,
            #         self.max_lr,
            #         self.min_lr,
            #     )
            #     stu_multiplier = [1.0, 1.0, 1.0, 1.0][pair_index] if pair_index < 4 else 0.1
            #     param_group["lr"] = stu_base_lr * stu_multiplier
            else:
                param_group["lr"] = self.get_lr(
                    relative_step,
                    self.warmup_steps,
                    self.num_steps,
                    self.max_lr,
                    self.min_lr,
                )

        self.optimizer.step()

        # Time how long this training step took
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time()
        dt = t1 - t0
        toks_processed = self.bsz * self.sl * self.world_size
        toks_per_sec = toks_processed / dt

        metrics = {
            "loss": loss.item(),
            "grad_norm": norm.item(),
            "step_time": dt,
            "tokens_per_sec": toks_per_sec,
        }

        # Add additional metrics if available
        if isinstance(loss_info, dict):
            metrics.update(
                {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in loss_info.items()
                }
            )

        return metrics

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model over an entire validation dataset.

        Args:
            dataloader (DataLoader): A DataLoader providing batches of data for evaluation.

        Returns:
            Dict[str, float]: A Dictionary of aggregated metrics over the dataset.
        """
        self.model.eval()
        val_steps = len(dataloader)
        metrics_accum = {"loss": 0.0, "tokens_processed": 0, "total_time": 0.0}
        additional_metrics = {}

        with (
            torch.no_grad(),
            tqdm(
                total=val_steps, desc="Validating", disable=not self.main_process
            ) as pbar,
        ):
            for inputs, targets in dataloader:
                t0 = time()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    preds, loss_info = self.model(inputs, targets)

                if isinstance(loss_info, tuple):
                    loss, *step_metrics = loss_info
                else:
                    loss = loss_info

                # Accumulate loss
                metrics_accum["loss"] += loss.item()

                # Accumulate additional metrics if available
                if isinstance(loss_info, dict):
                    for key, value in loss_info.items():
                        if key not in additional_metrics:
                            additional_metrics[key] = 0.0
                        additional_metrics[key] += (
                            value.item() if isinstance(value, torch.Tensor) else value
                        )

                # Time tracking
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time()
                dt = t1 - t0

                # Token processing tracking
                metrics_accum["tokens_processed"] += inputs.numel()
                metrics_accum["total_time"] += dt

                pbar.update(1)

        # Average the accumulated metrics
        metrics_avg = {
            "loss": metrics_accum["loss"] / val_steps,
            "tokens_per_sec": metrics_accum["tokens_processed"]
            / metrics_accum["total_time"],
        }

        # Average additional metrics
        for key, value in additional_metrics.items():
            metrics_avg[key] = value / val_steps

        # Synchronize metrics across processes if using distributed training
        if self.world_size > 1:
            for key in metrics_avg:
                dist.all_reduce(
                    torch.tensor(metrics_avg[key]).to(self.device), op=dist.ReduceOp.AVG
                )
                metrics_avg[key] = metrics_avg[key].item()

        return metrics_avg

    """
    Loss visualization method from Li et al. (2018).
    Adapted from https://github.com/nreHieW/loss/blob/main/main.py
    """

    def get_2_directions(self, verbose: bool = True):
        params = self.model.named_parameters()
        dx = {}
        dy = {}
        for name, param in params:
            curr_x = torch.randn_like(param)
            curr_y = torch.randn_like(param)
            if param.dim() <= 1:  # skip bias
                curr_x.fill_(0)
                curr_y.fill_(0)
            else:
                curr_x.mul_(param.norm() / (curr_x.norm() + 1e-10))
                curr_y.mul_(param.norm() / (curr_y.norm() + 1e-10))
            dx[name] = curr_x
            dy[name] = curr_y
        if verbose:
            _x = torch.cat([dx[name].flatten() for name in dx]).unsqueeze(0)
            _y = torch.cat([dy[name].flatten() for name in dy]).unsqueeze(0)
            similarity = F.cosine_similarity(_x, _y)
            print(f"cosine similarity between x-axis and y-axis: {similarity.item()}")
        return dx, dy

    def set_weights(
        self,
        model: nn.Module,
        original_state_dict: dict[str, torch.Tensor],
        dx: dict[str, torch.Tensor],
        dy: dict[str, torch.Tensor],
        x_step: float,
        y_step: float,
    ) -> nn.Module:
        for name, param in self.model.named_parameters():
            change = x_step * dx[name] + y_step * dy[name]
            param.data = original_state_dict[name].to(self.device) + change.to(
                self.device
            )

        return model

    def generate_loss_landscape(
        self, train_loader, output_path, x_range=(-1, 1, 2), y_range=(-1, 1, 2)
    ):
        original_state_dict = copy.deepcopy(self.model.state_dict())
        dx, dy = self.get_2_directions()

        x_min, x_max, x_num = x_range
        y_min, y_max, y_num = y_range

        x_coordinates = torch.linspace(x_min, x_max, x_num)
        y_coordinates = torch.linspace(y_min, y_max, y_num)

        total_iterations = x_num * y_num

        res = {}

        if self.main_process:
            pbar = tqdm(total=total_iterations, desc="Generating Loss Landscape")

        for i, x in enumerate(x_coordinates):
            for j, y in enumerate(y_coordinates):
                self.set_weights(self.model, original_state_dict, dx, dy, x, y)
                metrics = self.evaluate(train_loader)
                res[(i, j)] = (metrics["loss"], metrics.get("accuracy", 0.0))

                if self.main_process:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "x": f"{i+1}/{x_num}",
                            "y": f"{j+1}/{y_num}",
                            "coord": f"({x:.2f},{y:.2f})",
                            "loss": f"{metrics['loss']:.3f}",
                        }
                    )

        if self.main_process:
            pbar.close()

        # Restore original weights
        self.model.load_state_dict(original_state_dict)

        return res

    def convert_pt_to_vtk(
        self,
        pt_file_path,
        vtk_file_path,
        surf_name="train_loss",
        log=False,
        zmax=-1,
        interp=-1,
    ):
        # Load the PyTorch data
        data = torch.load(pt_file_path)

        i_coords = data["i_coords"].numpy()
        j_coords = data["j_coords"].numpy()
        losses = data[surf_name].numpy()

        # Create meshgrid
        xcoordinates, ycoordinates = np.meshgrid(
            np.unique(i_coords), np.unique(j_coords)
        )

        # Reshape losses to 2D
        vals = losses.reshape(xcoordinates.shape)

        # Interpolate if requested
        if interp > 0:
            m = interpolate.interp2d(
                xcoordinates[0, :], ycoordinates[:, 0], vals, kind="cubic"
            )
            x_array = np.linspace(xcoordinates.min(), xcoordinates.max(), interp)
            y_array = np.linspace(ycoordinates.min(), ycoordinates.max(), interp)
            z_array = m(x_array, y_array).ravel()
            x_array, y_array = np.meshgrid(x_array, y_array)
        else:
            x_array, y_array = xcoordinates, ycoordinates
            z_array = vals

        x_array = x_array.ravel()
        y_array = y_array.ravel()
        z_array = z_array.ravel()

        # Apply zmax if specified
        if zmax > 0:
            z_array = np.minimum(z_array, zmax)

        # Apply log scale if requested
        if log:
            z_array = np.log(z_array + 0.1)

        # Create a structured grid
        grid = vtkStructuredGrid()

        # Set dimensions
        grid.SetDimensions(len(np.unique(x_array)), len(np.unique(y_array)), 1)

        # Create points
        points = vtkPoints()
        for x, y, z in zip(x_array, y_array, z_array, strict=True):
            points.InsertNextPoint(x, y, z)
        grid.SetPoints(points)

        # Create a data array for the loss values
        loss_array = vtkDoubleArray()
        loss_array.SetName(surf_name)
        for z in z_array:
            loss_array.InsertNextValue(z)

        # Add the loss data to the grid
        grid.GetPointData().AddArray(loss_array)

        # Write the grid to a VTK file
        writer = vtkXMLStructuredGridWriter()
        writer.SetFileName(vtk_file_path)
        writer.SetInputData(grid)
        writer.Write()

        print(f"VTK file saved to {vtk_file_path}")

    def save_loss_landscape(
        self,
        landscape_data,
        output_path,
        convert_to_vtk=True,
        surf_name="train_loss",
        log=False,
        zmax=-1,
        interp=-1,
    ):
        """
        Save the loss landscape data and optionally convert it to VTK format.

        Args:
        - landscape_data: The loss landscape data dictionary
        - output_path: Path to save the PyTorch file
        - convert_to_vtk: Whether to also save as VTK file for ParaView (default: True)
        - surf_name: The type of surface to plot (default: 'train_loss')
        - log: Whether to use log scale for loss values in VTK (default: False)
        - zmax: Maximum z-value for capping in VTK (default: -1, no capping)
        - interp: Interpolate the surface to this resolution (default: -1, no interpolation)
        """
        # Convert the dictionary to tensors for more efficient storage
        i_coords, j_coords = zip(*landscape_data.keys(), strict=True)
        losses, accs = zip(*landscape_data.values(), strict=True)

        save_data = {
            "i_coords": torch.tensor(i_coords),
            "j_coords": torch.tensor(j_coords),
            "train_loss": torch.tensor(losses),
            "accuracy": torch.tensor(accs),
        }

        torch.save(save_data, output_path)
        if self.main_process:
            colored_print(f"Loss landscape saved to {output_path}", Colors.OKGREEN)

        if convert_to_vtk:
            vtk_path = output_path.replace(".pt", ".vtp")
            self.convert_pt_to_vtk(
                output_path,
                vtk_path,
                surf_name=surf_name,
                log=log,
                zmax=zmax,
                interp=interp,
            )
            if self.main_process:
                colored_print(f"VTK file saved to {vtk_path}", Colors.OKGREEN)

    def visualize_loss_landscape(self, landscape_data, output_path):
        if not self.main_process:
            return

        # Check if landscape_data is a file path or a dictionary
        if isinstance(landscape_data, str):
            # It's a file path, so load the data
            landscape_data = torch.load(landscape_data)
        elif not isinstance(landscape_data, dict):
            raise ValueError("landscape_data must be either a file path or a dictionary")

        i_coords = landscape_data["i_coords"].numpy()
        j_coords = landscape_data["j_coords"].numpy()
        losses = landscape_data["losses"].numpy()

        i_unique = sorted(set(i_coords))
        j_unique = sorted(set(j_coords))

        X, Y = np.meshgrid(i_unique, j_unique)
        Z = np.zeros_like(X, dtype=float)

        for idx, (i, j) in enumerate(zip(i_coords, j_coords, strict=True)):
            Z[j_unique.index(j), i_unique.index(i)] = losses[idx]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Use a perceptually uniform colormap
        cmap = plt.cm.viridis

        # Plot the surface with enhanced aesthetics
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            edgecolor="none",
            alpha=0.9,
            antialiased=True,
            shade=True,
            lightsource=LightSource(azdeg=315, altdeg=45),
        )

        # Customize the plot
        ax.set_xlabel("Direction 1", fontsize=14, labelpad=10)
        ax.set_ylabel("Direction 2", fontsize=14, labelpad=10)
        ax.set_zlabel("Loss", fontsize=14, labelpad=10)

        # Remove the background color
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Make the grid lines lighter
        ax.xaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 0.5)
        ax.yaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 0.5)
        ax.zaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 0.5)

        # Adjust the viewing angle for better visibility of convexity
        ax.view_init(elev=30, azim=135)

        # Add title with model details
        plt.title(
            f"Loss Landscape - {self.__class__.__name__}\n{self.optimizer.__class__.__name__}, Learning Rate: {self.max_lr}",
            fontsize=16,
            y=1.02,
        )

        # Add a color bar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
        cbar.ax.set_ylabel("Loss Value", rotation=270, labelpad=20, fontsize=12)

        # Tighten the layout and adjust margins
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save the figure with a higher DPI for better quality
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        print(f"Loss landscape visualization saved to {output_path}")
