import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import time
from typing import Callable, Tuple

from models.stu.model import SpectralSSM, SpectralSSMConfigs

from models.stu.benchmarks.synthetic import (
    generate_copy,
    generate_adding,
    generate_induction_heads,
    generate_associative_recall,
)


def benchmark_model(
    model: SpectralSSM,
    dataset: torch.utils.data.Dataset,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    task_name: str,
    batch_size: int = 32,
) -> Tuple[float, float]:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Task-specific input and target handling
            if task_name == "Copy Task":
                inputs = inputs.float()
                targets = targets.float()
            elif task_name == "Adding Task":
                inputs = inputs.float()
                targets = targets.float().unsqueeze(-1)
            elif task_name in ["Induction Heads", "Associative Recall"]:
                inputs = F.one_hot(
                    inputs.long(), num_classes=model.configs.d_in
                ).float()
                targets = F.one_hot(
                    targets.long(), num_classes=model.configs.d_out
                ).float()

            outputs, (loss, _) = model(inputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    exec_time = time.time() - start_time

    return avg_loss, exec_time


def run_benchmarks(model: SpectralSSM, configs: SpectralSSMConfigs):
    results = PrettyTable()
    results.field_names = ["Dataset", "Avg Loss", "Execution Time (s)"]

    datasets = [
        (
            "Copy Task",
            generate_copy(
                num_examples=1000,
                copy_len=10,
                blank_len=50,
                num_categories=configs.d_in,
            ),
        ),
        ("Adding Task", generate_adding(num_examples=1000, sequence_len=configs.sl)),
        (
            "Induction Heads",
            generate_induction_heads(
                num_examples=1000, sequence_len=configs.sl, vocab_size=configs.d_in
            ),
        ),
        (
            "Associative Recall",
            generate_associative_recall(
                num_examples=1000, sequence_len=configs.sl, vocab_size=configs.d_in
            ),
        ),
    ]

    for name, dataset in datasets:
        loss, exec_time = benchmark_model(model, dataset, configs.loss_fn, name)
        results.add_row([name, f"{loss:.4f}", f"{exec_time:.2f}"])

    print(results)


if __name__ == "__main__":
    # Initialize your model and configurations
    configs = SpectralSSMConfigs(
        d_in=128,  # Input dimension
        d_out=128,  # Output dimension
        sl=50,  # Sequence length
        n_layers=4,
        n_embd=128,
        num_eigh=16,
        k_u=2,
        k_y=2,
        use_hankel_L=True,
        learnable_m_y=True,
        alpha=0.5,
        bias=True,
        dropout=0.1,
    )

    model = SpectralSSM(configs)

    # Run benchmarks
    run_benchmarks(model, configs)
