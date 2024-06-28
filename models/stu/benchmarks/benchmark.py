# =============================================================================#
# Authors: Windsor Nguyen
# File: benchmark.py
# =============================================================================#

"""Benchmarking on synthetic long-context datasets."""

# TODO: This benchmarking code doesn't work quite yet because the model 
# dimensions were designed for CIFAR-10 and needs to be restructured.

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Architecture
from synthetic import (
    generate_copy,
    generate_adding,
    generate_induction_heads,
    generate_associative_recall,
)


class Benchmark:
    """
    Initializes and maintains the benchmarking state.
    """

    def __init__(
        self,
        model: Architecture,
        device: torch.device = None,
    ) -> None:
        """
        Initialize the benchmarking class.

        Args:
          model: The model to benchmark.
          device: The device to run the model on.
        """
        self.model = model
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, dataset_name: str, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model on a specific dataset.

        Args:
          dataset_name: Name of the dataset.
          dataloader: DataLoader for the dataset.

        Returns:
          A dictionary containing the average loss and accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                dataloader, desc=f'Evaluating on {dataset_name}', unit='batch'
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(-1)
                loss = self.criterion(outputs, targets)

                _, preds = torch.max(outputs, dim=1)
                correct = (preds == targets).sum().item()

                total_loss += loss.item() * targets.size(0)
                total_correct += correct
                total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples * 100

        return {'loss': avg_loss, 'accuracy': avg_accuracy}

    def benchmark(
        self, datasets: list[tuple[str, torch.utils.data.Dataset]], batch_size: int = 48
    ) -> None:
        """
        Benchmark the model on multiple datasets.

        Args:
          datasets: List of tuples containing dataset name and dataset.
          batch_size: Batch size for evaluation.
        """
        for dataset_name, dataset in datasets:
            dataloader = DataLoader(dataset, batch_size=batch_size)
            metrics = self.evaluate(dataset_name, dataloader)

            print(f'Dataset: {dataset_name}')
            print(f"  Average Loss: {metrics['loss']:.4f}")
            print(f"  Average Accuracy: {metrics['accuracy']:.2f}%")
            print()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Architecture(
        d_model=256,
        d_target=10,
        num_layers=6,
        dropout=0.1,
        input_len=32 * 32,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    ).to(device)
    checkpoint = torch.load('../checkpoint.pt', map_location=device)
    model.load_state_dict(checkpoint)

    datasets = [
        (
            'Copy',
            generate_copy(
                num_examples=1000, num_categories=10, copy_len=10, blank_len=5
            ),
        ),
        ('Adding', generate_adding(num_examples=1000, sequence_len=10)),
        (
            'Induction Heads',
            generate_induction_heads(num_examples=1000, sequence_len=30, vocab_size=20),
        ),
        (
            'Associative Recall',
            generate_associative_recall(
                num_examples=1000, sequence_len=30, vocab_size=10
            ),
        ),
    ]

    benchmark = Benchmark(model, device)
    benchmark.benchmark(datasets)


if __name__ == '__main__':
    main()
