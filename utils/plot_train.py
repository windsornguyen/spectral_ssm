# =============================================================================#
# Authors: Windsor Nguyen
# File: plot.py
# =============================================================================#

"""A malleable plotting script."""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.colors import Colors, colored_print
from scipy.ndimage import gaussian_filter1d


def load_data(file_path):
    while True:
        try:
            with open(file_path, "r") as f:
                data = [float(line.strip()) for line in f if line.strip()]
            return np.array(data)
        except FileNotFoundError:
            colored_print(f"File not found: {file_path}", Colors.FAIL)
            file_path = input(f"Please enter a valid file path for {file_path}: ")
        except ValueError:
            colored_print(f"Invalid data format in file: {file_path}", Colors.FAIL)
            file_path = input(f"Please enter a valid file path for {file_path}: ")


def apply_gaussian_smoothing(data: np.ndarray, sigma: float = 2) -> np.ndarray:
    return gaussian_filter1d(data, sigma)


def plot_data(
    data_list: list[np.ndarray],
    time_steps_list: list[np.ndarray],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes = None,
    smoothing: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for data, time_steps, label in zip(data_list, time_steps_list, labels, strict=True):
        if smoothing:
            data = apply_gaussian_smoothing(data)
        ax.plot(time_steps, data, linewidth=2, label=label)

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12, loc="best", frameon=True, fancybox=True, shadow=True)

    # Improve aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    fig.tight_layout()
    return fig, ax


def get_user_input(prompt, data_type=str):
    while True:
        user_input = input(prompt)
        try:
            return data_type(user_input)
        except ValueError:
            colored_print("Invalid input. Please try again.", Colors.WARNING)


def main():
    colored_print("Welcome to the enhanced plotting script!", Colors.HEADER)

    plot_data_list = []
    plot_time_steps_list = []
    plot_labels = []
    plot_titles = []
    plot_xlabels = []
    plot_ylabels = []

    while True:
        colored_print("\nWhat do you want to plot today?", Colors.OKBLUE)
        colored_print("1. Train losses", Colors.OKGREEN)
        colored_print("2. Validation losses", Colors.OKGREEN)
        colored_print("3. Test losses", Colors.OKGREEN)
        colored_print("4. Train and val", Colors.OKGREEN)
        colored_print("5. Gradient norms", Colors.OKGREEN)
        colored_print("6. Other files together", Colors.OKGREEN)

        choice = get_user_input(f"{Colors.BOLD}Enter your choice (1-6):{Colors.ENDC} ")

        if choice in ["1", "2", "3", "5", "6"]:
            num_plots = get_user_input(
                f"{Colors.BOLD}Enter the number of plots:{Colors.ENDC} ", int
            )
            data_list = []
            time_steps_list = []
            labels = []
            for i in range(num_plots):
                file_path = input(f"Enter the file path for plot {i+1} (.txt): ")
                data = load_data(file_path)
                if data is not None:
                    data_list.append(data)
                    label = input(f"Enter the legend name for plot {i+1}: ")
                    labels.append(label)

                    if "val" in file_path.lower():
                        time_steps_file_path = input(
                            f"Enter the file path for validation time steps plot {i+1} (.txt): "
                        )
                        time_steps_data = load_data(time_steps_file_path)
                        if time_steps_data is not None:
                            time_steps_list.append(time_steps_data)
                        else:
                            colored_print(
                                f"Validation time steps data not found for plot {i+1}. Using default steps.",
                                Colors.WARNING,
                            )
                            time_steps_list.append(np.arange(len(data)))
                    else:
                        time_steps_list.append(np.arange(len(data)))

            title = input("Enter the title for the plot: ")

            if choice == "1":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title or "Training Losses")
                plot_xlabels.append("Epochs")
                plot_ylabels.append("Loss")
            elif choice == "2":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title or "Validation Losses")
                plot_xlabels.append("Time Steps")
                plot_ylabels.append("Loss")
            elif choice == "3":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title or "Test Losses")
                plot_xlabels.append("Epochs")
                plot_ylabels.append("Loss")
            elif choice == "5":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title or "Gradient Norms")
                plot_xlabels.append("Steps")
                plot_ylabels.append("Norm")
            elif choice == "6":
                plot_data_list.append(data_list)
                plot_time_steps_list.append(time_steps_list)
                plot_labels.append(labels)
                plot_titles.append(title or "Custom Plot")
                plot_xlabels.append(input("Enter x-axis label: "))
                plot_ylabels.append(input("Enter y-axis label: "))

        elif choice == "4":
            num_train_plots = get_user_input(
                f"{Colors.BOLD}Enter the number of train plots:{Colors.ENDC} ", int
            )
            train_data_list = []
            train_time_steps_list = []
            train_labels = []
            for i in range(num_train_plots):
                file_path = input(f"Enter the file path for train plot {i+1}: ")
                data = load_data(file_path)
                if data is not None:
                    train_data_list.append(data)
                    label = input(f"Enter the legend name for train plot {i+1}: ")
                    train_labels.append(label)
                    train_time_steps_list.append(np.arange(len(data)))

            num_val_plots = get_user_input(
                f"{Colors.BOLD}Enter the number of val plots:{Colors.ENDC} ", int
            )
            val_data_list = []
            val_time_steps_list = []
            val_labels = []
            for i in range(num_val_plots):
                file_path = input(f"Enter the file path for val plot {i+1}: ")
                data = load_data(file_path)
                if data is not None:
                    val_data_list.append(data)
                    label = input(f"Enter the legend name for val plot {i+1}: ")
                    val_labels.append(label)

                    time_steps_file_path = input(
                        f"Enter the file path for validation time steps plot {i+1}: "
                    )
                    time_steps_data = load_data(time_steps_file_path)
                    if time_steps_data is not None:
                        val_time_steps_list.append(time_steps_data)
                    else:
                        colored_print(
                            f"Validation time steps data not found for plot {i+1}. Using default steps.",
                            Colors.WARNING,
                        )
                        val_time_steps_list.append(np.arange(len(data)))

            plot_data_list.append(train_data_list + val_data_list)
            plot_time_steps_list.append(train_time_steps_list + val_time_steps_list)
            plot_labels.append(train_labels + val_labels)
            title = input("Enter the title for the training and validation plot: ")
            plot_titles.append(title or "Training and Validation Losses")
            plot_xlabels.append("Steps")
            plot_ylabels.append("Loss")

        else:
            colored_print("Invalid choice. Please try again.", Colors.WARNING)
            continue

        more_plots = input("Do you want more plots? (y/n): ").lower()
        if more_plots != "y":
            break

    same_plot = input(
        "Do you want to plot all graphs on the same plot? (y/n): "
    ).lower()
    smoothing = input("Do you want to apply Gaussian smoothing? (y/n): ").lower() == "y"

    save_path = input("Which directory should the plot be saved in?: ")
    os.makedirs(save_path, exist_ok=True)

    if same_plot == "y":
        fig, ax = plt.subplots(figsize=(12, 8))
        for data_list, time_steps_list, labels, title, xlabel, ylabel in zip(
            plot_data_list,
            plot_time_steps_list,
            plot_labels,
            plot_titles,
            plot_xlabels,
            plot_ylabels,
            strict=True,
        ):
            plot_data(
                data_list, time_steps_list, labels, title, xlabel, ylabel, ax, smoothing
            )

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"combined_plot_{current_time}.png"
        filename = (
            input(
                f"Enter the file name for the combined plot (default: {default_filename}): "
            )
            or default_filename
        )
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
        colored_print(
            f"Combined plot saved at: {os.path.join(save_path, filename)}",
            Colors.OKGREEN,
        )
    else:
        for i, (data_list, time_steps_list, labels, title, xlabel, ylabel) in enumerate(
            zip(
                plot_data_list,
                plot_time_steps_list,
                plot_labels,
                plot_titles,
                plot_xlabels,
                plot_ylabels,
                strict=True,
            )
        ):
            fig, ax = plot_data(
                data_list,
                time_steps_list,
                labels,
                title,
                xlabel,
                ylabel,
                smoothing=smoothing,
            )
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_filename = f"plot_{i+1}_{current_time}.png"
            filename = (
                input(
                    f"Enter the filename for plot {i+1} (default: {default_filename}): "
                )
                or default_filename
            )
            plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
            colored_print(
                f"Plot {i+1} saved at: {os.path.join(save_path, filename)}",
                Colors.OKGREEN,
            )

    plt.show()
    colored_print("Plotting completed successfully!", Colors.OKGREEN)


if __name__ == "__main__":
    main()
