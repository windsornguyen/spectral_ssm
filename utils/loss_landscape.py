# ==============================================================================#
# Authors: Windsor Nguyen
# File: loss_landscape.py
# ==============================================================================#

import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LogNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from vtk import (
    vtkStructuredGrid,
    vtkPoints,
    vtkFloatArray,
    vtkXMLStructuredGridWriter,
    vtkLookupTable,
    vtkWarpScalar,
)
from utils.colors import Colors, colored_print


class LossLandscape:
    """
    Loss visualization method from Li et al.,
    "Visualizing the Loss Landscape of Neural Nets" (NeurIPS, 2018).

    Adapted from https://github.com/nreHieW/loss/blob/main/main.py/
    Original repository: https://github.com/tomgoldstein/loss-landscape/
    """

    def __init__(self, model, device, optimizer, max_lr, main_process: bool = False, dtype=torch.float32):
        self.model = model
        self.device = device
        # self.device = torch.device(
        #     "cpu"
        # )  # TODO: Verify that this class works on CPU devices as well.
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.main_process = main_process
        self.dtype = dtype
        self.model.to(self.device)

    def generate_loss_landscape(
        self,
        dataloader: DataLoader,
        x_range: tuple[float, float, int] = (-1, 1, 10),
        y_range: tuple[float, float, int] = (-1, 1, 10),
        batch_size: int = 8,
    ) -> dict[tuple[float, float], float]:
        """
        Generates the loss landscape of a model over a range of perturbations
        applied to the model's parameters.

        Args:
            dataloader (DataLoader): The evaluation dataloader.
            x_range (tuple[float, float, int]): The range of x values (min, max, num_points).
            y_range (tuple[float, float, int]): The range of y values (min, max, num_points).
            batch_size (int): Number of points to evaluate in parallel (for GPU processing).

        Returns:
            dict[tuple[float, float], float]: A dictionary with keys
                as coordinate tuples (x, y) and values as tuples of
                (loss, accuracy) at the corresponding perturbation points.
        """
        original_params = copy.deepcopy(self.model.state_dict())
        dx, dy = self._get_perturbation()

        x_min, x_max, x_num = x_range
        y_min, y_max, y_num = y_range
        x_coords = torch.linspace(x_min, x_max, x_num)
        y_coords = torch.linspace(y_min, y_max, y_num)

        coordinates = [(x.item(), y.item()) for x in x_coords for y in y_coords]
        total_points = len(coordinates)

        with tqdm(
            total=total_points,
            desc="Generating Loss Landscape",
            disable=not self.main_process,
        ) as pbar:
            if torch.cuda.is_available():
                results = self._generate_cuda(
                    dataloader, coordinates, dx, dy, original_params, batch_size, pbar
                )
            else:
                results = self._generate_cpu(
                    dataloader, coordinates, dx, dy, original_params, pbar
                )

        # Restore the original model parameters
        self.model.load_state_dict(original_params)

        return results

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> float:
        """
        Evaluate the model over the evaluation dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (DataLoader): A DataLoader providing batches of data for evaluation.

        Returns:
            float: The average loss over the evaluation dataset.
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device).to(self.dtype)
                targets = targets.to(self.device).to(self.dtype)
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    _, loss_info = model(inputs, targets)

                if isinstance(loss_info, tuple):
                    loss, *step_metrics = loss_info
                else:
                    loss = loss_info

                total_loss += loss.item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def _generate_cuda(
        self,
        dataloader: DataLoader,
        coords: list[tuple[float, float]],
        dx: dict[str, torch.Tensor],
        dy: dict[str, torch.Tensor],
        original_params: dict[str, torch.Tensor],
        batch_size: int,
        pbar: tqdm,
    ) -> dict[tuple[float, float], float]:
        """
        Generates the loss landscape using CUDA-enabled GPUs.

        Args:
            dataloader (DataLoader): The training dataloader.
            coords (list): List of (x, y) coordinate tuples to evaluate.
            dx (dict): The x-direction perturbation.
            dy (dict): The y-direction perturbation.
            original_params (dict): The original model parameters.
            batch_size (int): Number of points to evaluate in parallel.

        Returns:
            dict: A dictionary with keys as coordinate tuples (x, y) and values
                as tuples of (loss, accuracy) at the corresponding perturbation points.
        """
        results = {}
        num_batches = math.ceil(len(coords) / batch_size)

        for i in range(num_batches):
            batch_coords = coords[i * batch_size : (i + 1) * batch_size]
            batch_models = [
                copy.deepcopy(self.model).to(self.device).to(self.dtype) for _ in batch_coords
            ]

            for model, (X, y) in zip(batch_models, batch_coords, strict=True):
                self._apply_perturbation(model, original_params, dx, dy, X, y)

            batch_results = [self.evaluate(model, dataloader) for model in batch_models]

            for (X, y), loss in zip(batch_coords, batch_results, strict=True):
                results[(X, y)] = loss

            pbar.update(len(batch_coords))

        return results

    def _generate_cpu(
        self,
        dataloader: DataLoader,
        coords: list[tuple[float, float]],
        dx: dict[str, torch.Tensor],
        dy: dict[str, torch.Tensor],
        original_params: dict[str, torch.Tensor],
        pbar: tqdm,
    ) -> dict[tuple[float, float], float]:
        """
        Generates the loss landscape using CPU multiprocessing.

        Args:
            dataloader (DataLoader): The evaluation dataloader.
            coords (list): List of (x, y) coordinate tuples to evaluate.
            dx (dict): The x-direction perturbation.
            dy (dict): The y-direction perturbation.
            original_params (dict): The original model parameters.
            pbar (tqdm): Progress bar object for tracking.

        Returns:
            dict: A dictionary with keys as coordinate tuples (x, y) and values
                as loss at the corresponding perturbation points.
        """
        num_processes = mp.cpu_count()
        chunk_size = math.ceil(len(coords) / num_processes)
        chunks = [coords[i : i + chunk_size] for i in range(0, len(coords), chunk_size)]

        def process_wrapper(chunk):
            return self._process_chunk(chunk, dataloader, dx, dy, original_params)

        results = {}
        for chunk_result in process_map(
            process_wrapper, chunks, max_workers=num_processes, disable=True
        ):
            results.update(chunk_result)
            pbar.update(len(chunk_result))

        return results

    def _process_chunk(
        self,
        chunk: list[tuple[float, float]],
        dataloader: DataLoader,
        dx: dict[str, torch.Tensor],
        dy: dict[str, torch.Tensor],
        original_params: dict[str, torch.Tensor],
    ) -> dict[tuple[float, float], float]:
        """
        Processes a chunk of coordinates for CPU-based loss landscape generation.

        Args:
            chunk (list): A subset of coordinates to process.
            dataloader (DataLoader): The evaluation dataloader.
            dx (dict): The x-direction perturbation.
            dy (dict): The y-direction perturbation.
            original_params (dict): The original model parameters.

        Returns:
            dict: A dictionary with keys as coordinate tuples (x, y) and values
                as loss for the processed chunk.
        """
        results = {}
        model = copy.deepcopy(self.model)
        for X, y in chunk:
            self._apply_perturbation(model, original_params, dx, dy, X, y)
            loss = self.evaluate(model, dataloader)
            results[(X, y)] = loss
        return results

    def _get_perturbation(self, verbose: bool = True):
        """
        Creates two random directions dx and dy in the parameter space,
        normalized to have the same norm as the corresponding model parameters.

        The goal is to generate random perturbations to analyze the effect of
        small changes in the parameter space.

        Args:
            verbose (bool, optional): Whether to print the norms of the directions. Defaults to True.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: Two dictionaries of directions dx and dy.
        """
        params = self.model.named_parameters()
        dx, dy = {}, {}

        for name, param in params:
            # Generate a random perturbation tensor w/ same shape as the parameter
            ptbd_x = torch.randn_like(param)
            ptbd_y = torch.randn_like(param)

            if param.dim() <= 1:  # Skip biases (apply no perturbation)
                ptbd_x.fill_(0)
                ptbd_y.fill_(0)
            else:
                ptbd_x = F.normalize(ptbd_x, p=2, dim=None) * param.norm()
                ptbd_y = F.normalize(ptbd_y, p=2, dim=None) * param.norm()

            # Save the perturbation to its respective parameter
            dx[name] = ptbd_x
            dy[name] = ptbd_y

        if verbose:
            _x = torch.cat([dx[name].flatten() for name in dx]).unsqueeze(0)
            _y = torch.cat([dy[name].flatten() for name in dy]).unsqueeze(0)
            similarity = F.cosine_similarity(_x, _y)
            if self.main_process:
                print(
                    f"Cosine similarity between x-axis and y-axis: {similarity.item()}"
                )

        return dx, dy

    def _apply_perturbation(
        self,
        model: torch.nn.Module,
        param: dict[str, torch.Tensor],
        dx: dict[str, torch.Tensor],
        dy: dict[str, torch.Tensor],
        x_step: float,
        y_step: float,
    ) -> nn.Module:
        """
        Applies perturbations to a given model's parameters.

        Args:
            model (nn.Module): The model to perturb.
            param (dict[str, torch.Tensor]): The original state dict of the model.
            dx (dict[str, torch.Tensor]): The perturbation in the x direction.
            dy (dict[str, torch.Tensor]): The perturbation in the y direction.
            x_step (float): The step size in the x direction.
            y_step (float): The step size in the y direction.

        Returns:
            nn.Module: The perturbed model.
        """
        for name, updated_param in model.named_parameters():
            perturbed_param = x_step * dx[name] + y_step * dy[name]
            updated_param.data = (param[name].to(self.device) + perturbed_param.to(self.device)).to(self.dtype)

        return model

    def convert_to_vts(
        self,
        landscape_data: str,
        vts_file_path: str,
        surface_name: str = "loss",
        apply_log: bool = False,
        zmax: float = -1,
        interp_size: int = 100,
        grid_density: int = 10,
    ) -> None:
        """
        Converts loss landscape data to VTS file format for visualization in ParaView.

        Args:
            landscape_data (str): Path to the saved loss landscape data.
            vts_file_path (str): Path to save the output VTS file.
            surface_name (str, optional): Name of the surface data. Defaults to "loss".
            apply_log (bool, optional): Apply log transformation to the data. Defaults to False.
            zmax (float, optional): Maximum z-value cap. If > 0, clamps data to this value. Defaults to -1 (no capping).
            interp_size (int, optional): Size of the interpolation grid. Defaults to 100.
            grid_density (int, optional): Density of the grid lines. Defaults to 10.
        """
        try:
            # Load the saved landscape data
            saved_data = torch.load(landscape_data)
            x_coords = saved_data["x_coords"]
            y_coords = saved_data["y_coords"]
            losses = saved_data["loss"]

            # Create meshgrid
            xs, ys = torch.meshgrid(
                torch.unique(x_coords), torch.unique(y_coords), indexing="ij"
            )

            # Reshape losses to 2D
            vals = losses.reshape(xs.shape)

            # Interpolate
            x_interp = torch.linspace(xs.min(), xs.max(), interp_size)
            y_interp = torch.linspace(ys.min(), ys.max(), interp_size)
            grid_x, grid_y = torch.meshgrid(x_interp, y_interp, indexing="ij")

            # Use grid_sample for interpolation
            vals_expanded = vals.unsqueeze(0).unsqueeze(0)
            grid_normalized = torch.stack(
                [
                    2 * (grid_x - xs.min()) / (xs.max() - xs.min()) - 1,
                    2 * (grid_y - ys.min()) / (ys.max() - ys.min()) - 1,
                ],
                dim=-1,
            ).unsqueeze(0)

            interpolated_vals = torch.nn.functional.grid_sample(
                vals_expanded, grid_normalized, align_corners=True, mode="bicubic"
            ).squeeze()

            # Apply zmax if specified
            if zmax > 0:
                interpolated_vals = torch.clamp(interpolated_vals, max=zmax)

            # Apply log scale if requested
            if apply_log:
                interpolated_vals = torch.log(interpolated_vals + 0.1)

            # Create a structured grid
            grid = vtkStructuredGrid()
            grid.SetDimensions(interp_size, interp_size, 1)

            # Create points
            points = vtkPoints()
            for X, y, _ in zip(grid_x.ravel(), grid_y.ravel(), interpolated_vals.ravel(), strict=True):
                points.InsertNextPoint(X.item(), y.item(), 0)  # Set z to 0 initially
            grid.SetPoints(points)

            # Create a data array for the loss values
            loss_array = vtkFloatArray()
            loss_array.SetName(surface_name)
            for z in interpolated_vals.ravel():
                loss_array.InsertNextValue(z.item())

            # Add the loss data to the grid
            grid.GetPointData().AddArray(loss_array)
            grid.GetPointData().SetActiveScalars(surface_name)

            # Use vtkWarpScalar to create the 3D surface
            warp = vtkWarpScalar()
            warp.SetInputData(grid)
            warp.SetScaleFactor(1)  # Adjust this value to control the height of the surface
            warp.Update()

            # Create a color lookup table
            lut = vtkLookupTable()
            lut.SetHueRange(0.0, 0.667)  # Red to Blue
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.SetTableRange(interpolated_vals.min().item(), interpolated_vals.max().item())
            lut.Build()

            # Create a grid/wireframe representation
            grid_array = vtkFloatArray()
            grid_array.SetName("GridLines")
            for i in range(interp_size * interp_size):
                x = i % interp_size
                y = i // interp_size
                if x % grid_density == 0 or y % grid_density == 0:
                    grid_array.InsertNextValue(1)  # Grid line
                else:
                    grid_array.InsertNextValue(0)  # Not a grid line

            # Add the grid data to the warped grid
            warp.GetOutput().GetPointData().AddArray(grid_array)

            # Write the warped grid to a VTS file
            writer = vtkXMLStructuredGridWriter()
            writer.SetFileName(vts_file_path)
            writer.SetInputData(warp.GetOutput())
            writer.Write()

            colored_print(f"VTS file with grid saved to {vts_file_path}", Colors.OKGREEN)
        except Exception as e:
            colored_print(f"Error occurred while saving enhanced VTS file: {str(e)}", Colors.FAIL)

    def save_loss_landscape(
        self,
        landscape_data: dict[tuple[float, float], float],
        output_path: str,
        convert_to_vts: bool = True,
        surface_name: str = "loss",
        apply_log: bool = False,
        zmax: float = -1,
        interp_size: int = 100,
    ) -> None:
        """
        Saves the loss landscape data and optionally convert it to VTS format.

        Args:
            landscape_data (Dict[Tuple[float, float], float]): The loss landscape data dictionary.
            output_path (str): Path to save the PyTorch file.
            convert_to_vts (bool, optional): Whether to also save as VTS file for ParaView. Defaults to True.
            surface_name (str, optional): The type of surface to plot. Defaults to 'loss'.
            apply_log (bool, optional): Whether to use log scale for loss values in VTS. Defaults to False.
            zmax (float, optional): Maximum z-value for capping in VTS. If > 0, clamps data to this value. Defaults to -1 (no capping).
            interp_size (int, optional): Interpolate the surface to this resolution. Defaults to 100.

        Returns:
            None
        """
        try:
            # Convert the dictionary to tensors for more efficient storage
            coords, losses = zip(*landscape_data.items(), strict=True)
            x_coords, y_coords = zip(*coords, strict=True)

            save_data = {
                "x_coords": torch.tensor(x_coords),
                "y_coords": torch.tensor(y_coords),
                "loss": torch.tensor(losses),
            }

            torch.save(save_data, output_path)
            if self.main_process:
                colored_print(f"Loss landscape saved to {output_path}", Colors.OKGREEN)

            if convert_to_vts:
                vts_path = output_path.replace(".pt", ".vts")
                self.convert_to_vts(
                    output_path,
                    vts_path,
                    surface_name=surface_name,
                    apply_log=apply_log,
                    zmax=zmax,
                    interp_size=interp_size,
                )
        except Exception as e:
            colored_print(
                f"Error occurred while saving loss landscape: {str(e)}", Colors.FAIL
            )

    def visualize_loss_landscape(
        self,
        landscape_data: str,
        output_path: str,
    ):
        """
        Visualizes the loss landscape and save it as an image.

        Args:
            landscape_data (str): Path to the saved loss landscape data.
            output_path (str): Path to save the output image.

        Returns:
            None
        """
        try:
            # Load the saved landscape data
            saved_data = torch.load(landscape_data)
            x_coords = saved_data["x_coords"].numpy()
            y_coords = saved_data["y_coords"].numpy()
            losses = saved_data["loss"].numpy()

            x_unique = sorted(set(x_coords))
            y_unique = sorted(set(y_coords))

            X, Y = np.meshgrid(x_unique, y_unique)
            Z = np.zeros_like(X, dtype=float)

            for x, y, loss in zip(x_coords, y_coords, losses, strict=True):
                Z[y_unique.index(y), x_unique.index(x)] = loss

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
            ax.set_xlabel("X", fontsize=14, labelpad=10)
            ax.set_ylabel("Y", fontsize=14, labelpad=10)
            ax.set_zlabel("Loss Value", fontsize=14, labelpad=10)

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
            optimizer_name = (
                self.optimizer.__class__.__name__ if self.optimizer else "N/A"
            )
            lr_info = f", Learning Rate: {self.max_lr}" if self.max_lr else ""
            plt.title(
                f"Loss Landscape - {self.model.__class__.__name__}\n{optimizer_name}{lr_info}",
                fontsize=16,
                y=1.02,
            )

            # Add a color bar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
            cbar.ax.set_ylabel("Loss Value", rotation=270, labelpad=20, fontsize=12)

            # Tighten the layout and adjust margins
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            # Save the figure with a high DPI for better quality
            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close()

            colored_print(
                f"Loss landscape visualization saved to {output_path}", Colors.OKGREEN
            )
        except Exception as e:
            colored_print(
                f"Error occurred while visualizing loss landscape: {str(e)}",
                Colors.FAIL,
            )

    def plot_hessian_heatmap(self, landscape_data: str, output_path: str):
        """
        Computes and plots a heatmap of the ratio of minimum to maximum Hessian
        eigenvalues to visualize local convexity in the loss landscape.

        We can measure convexity by computing the principal curvatures, which are
        the eigenvalues of the Hessian matrix. A convex function has non-negative
        curvatures, i.e. a positive-semidefinite Hessian.

        Note: Non-convexity in the dimensionality-reduced plot implies
        non-convexity in the full-dimensional surface, but "apparent" convexity
        in the dimensionality-reduced plot does not mean the high-dimensional
        surface is truly convex. Rather, it implies that the positive curvatures
        are dominant. More formally, the _mean_ curvature (average eigenvalue)
        is positive.

        Args:
            landscape_data (str): Path to the saved loss landscape data.
            output_path (str): Path to save the output heatmap image.

        Returns:
            None
        """
        try:
            # Load the saved landscape data
            saved_data = torch.load(landscape_data)
            x_coords = saved_data["x_coords"]
            y_coords = saved_data["y_coords"]
            losses = saved_data["loss"]

            # Create unique sorted coordinates
            x_unique, x_indices = torch.unique(
                x_coords, sorted=True, return_inverse=True
            )
            y_unique, y_indices = torch.unique(
                y_coords, sorted=True, return_inverse=True
            )

            # Create the loss grid
            Z = torch.zeros((len(y_unique), len(x_unique)), dtype=torch.float32)
            Z[y_indices, x_indices] = losses

            # Compute gradients and Hessian
            dy, dx = torch.gradient(Z)
            dxx, dxy = torch.gradient(dx)
            _, dyy = torch.gradient(dy)

            # Compute eigenvalues of the Hessian
            det = dxx * dyy - dxy**2
            trace = dxx + dyy
            sqrt_term = torch.sqrt(torch.abs(trace**2 - 4 * det))
            max_eigenvalues = (trace + sqrt_term) / 2
            min_eigenvalues = (trace - sqrt_term) / 2

            # Compute the ratio of minimum to maximum eigenvalues
            ratio = torch.abs(min_eigenvalues / max_eigenvalues)
            ratio = ratio.numpy()

            # Plot the heatmap
            plt.figure(figsize=(12, 10))
            plt.imshow(
                ratio,
                extent=[
                    x_unique[0].item(),
                    x_unique[-1].item(),
                    y_unique[0].item(),
                    y_unique[-1].item(),
                ],
                origin="lower",
                cmap="RdYlBu_r",  # Yellow-Orange-Blue colormap, reversed
                aspect="auto",
                norm=LogNorm(
                    vmin=max(ratio.min(), 1e-8), vmax=1
                ),  # Log scale for better visualization
            )
            plt.colorbar(label="|λ_min / λ_max|")
            plt.title("Hessian Eigenvalue Ratios")
            plt.xlabel("X Axis")
            plt.ylabel("Y Axis")

            # Add contour lines of the loss landscape
            X, Y = torch.meshgrid(x_unique, y_unique, indexing="xy")
            contour = plt.contour(
                X.numpy(), Y.numpy(), Z.numpy(), colors="k", alpha=0.3
            )
            plt.clabel(contour, inline=True, fontsize=8)

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            colored_print(
                f"Hessian eigenvalue ratio heatmap saved to {output_path}",
                Colors.OKGREEN,
            )

            # Calculate some statistics
            convex_ratio = (ratio >= 0).mean()
            near_convex_ratio = (ratio >= -0.01).mean()
            print(f"Percentage of convex regions: {convex_ratio:.2%}")
            print(
                f"Percentage of near-convex regions (ratio >= -0.01): {near_convex_ratio:.2%}"
            )

        except Exception as e:
            colored_print(
                f"Error occurred while analyzing Hessian eigenvalue ratio: {str(e)}",
                Colors.FAIL,
            )

    def generate(
        self,
        dataloader: DataLoader,
        output_path: str,
        x_range: tuple[float, float, int] = (-1, 1, 10),
        y_range: tuple[float, float, int] = (-1, 1, 10),
        batch_size: int = 10,
        convert_to_vts: bool = True,
        surface_name: str = "loss",
        apply_log: bool = False,
        zmax: float = -1,
        interp_size: int = 100,
        plot_loss_landscape: bool = True,
        plot_hessian: bool = True,
    ) -> None:
        """
        Generates the loss landscape, saves it as a PyTorch file, converts to VTS format,
        creates a visualization, and optionally plots a Hessian heatmap.

        Args:
            dataloader (DataLoader): The evaluation dataloader.
            output_path (str): Base path for output files.
            x_range (Tuple[float, float, int]): The range of x values (min, max, num_points).
            y_range (Tuple[float, float, int]): The range of y values (min, max, num_points).
            batch_size (int): Number of points to evaluate in parallel (for GPU processing).
            convert_to_vts (bool): Whether to also save as VTS file for ParaView.
            surface_name (str): Name of the surface data in the VTS file.
            apply_log (bool): Apply log transformation to the data.
            zmax (float): Maximum z-value cap. If > 0, clamps data to this value.
            interp_size (int): Size of the interpolation grid for VTS conversion.
            plot_loss_landscape (bool): Whether to plot the loss landscape.
            plot_hessian (bool): Whether to plot the heatmap of the Hessian eigenvalue ratios.
        """
        landscape_data = self.generate_loss_landscape(
            dataloader, x_range, y_range, batch_size
        )
        pt_path = f"{output_path}.pt"
        self.save_loss_landscape(
            landscape_data,
            pt_path,
            convert_to_vts,
            surface_name,
            apply_log,
            zmax,
            interp_size,
        )
        if plot_loss_landscape:
            self.visualize_loss_landscape(pt_path, f"{output_path}.png")

        if plot_hessian:
            self.plot_hessian_heatmap(pt_path, f"{output_path}_hessian_heatmap.png")
