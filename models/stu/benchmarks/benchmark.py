# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: benchmark.py
# =============================================================================#

"""Benchmarking on synthetic long-context datasets."""

import argparse
import torch
from models.stu.benchmarks.model import SpectralSSM, SpectralSSMConfigs
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from utils.colors import Colors, colored_print
from utils.dist import setup, cleanup
from models.stu.benchmarks.synthetic import (
    generate_copy,
    generate_adding,
    generate_induction_heads,
    generate_associative_recall,
)
    
def get_dataloader(dataset, bsz, shuffle=True, distributed=False, **kwargs):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs
    )

class Benchmark:
    """
    A class for benchmarking the Spectral SSM model on various tasks.
    """
    def __init__(self, model, device=None):
        """
        Initialize the Benchmark class.

        Args:
            model (torch.nn.Module): The model to benchmark.
            device (torch.device, optional): The device to run the model on.
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)

    def train(self, dataset, num_epochs=10, batch_size=48):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = self.model(inputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    def evaluate(self, dataset_name, dataloader):
        """
        Evaluate the model on a given dataset.

        Args:
            dataset_name (str): Name of the dataset.
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            dict: Dictionary containing the average loss and accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f"Evaluating on {dataset_name}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, loss = self.model(inputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, dim=-1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.numel()

        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100
        return avg_loss, accuracy

    def benchmark(self, datasets, num_epochs=10, batch_size=48):
        """
        Run the benchmark on multiple datasets.

        Args:
            datasets (list): List of tuples containing dataset name and dataset.
            batch_size (int): Batch size for evaluation.
        """
        for dataset_name, dataset in datasets:
            # First, train the model
            self.train(dataset, num_epochs=num_epochs, batch_size=batch_size)

            # Then evaluate
            dataloader = DataLoader(dataset, batch_size=batch_size)
            loss, accuracy = self.evaluate(dataset_name, dataloader)

            colored_print(f"Dataset: {dataset_name}", Colors.HEADER)
            colored_print(f"  Average Loss: {loss:.4f}", Colors.OKBLUE)
            colored_print(f"  Accuracy: {accuracy:.2f}%", Colors.OKGREEN)
            print()

def get_model(args, device):
    """
    Create and configure the Spectral SSM model.

    Args:
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device to run the model on.

    Returns:
        SpectralSSM: Configured Spectral SSM model.
    """
    configs = SpectralSSMConfigs(
        n_layers=args.n_layers,
        n_embd=args.num_categories,
        d_in=args.num_categories,
        d_out=args.num_categories,
        d_proj=args.num_categories,
        sl=args.sl,
        scale=args.scale,
        bias=args.bias,
        dropout=args.dropout,
        num_eigh=args.num_eigh,
        k_y=args.k_y,
        k_u=args.k_u,
        learnable_m_y=args.learnable_m_y,
        alpha=args.alpha,
        use_ar_y=args.use_ar_y,
        use_ar_u=args.use_ar_u,
        use_hankel_L=args.use_hankel_L,
        moe=args.moe,
        num_experts=args.num_experts,
        num_experts_per_timestep=args.num_experts_per_timestep,
        loss_fn=CrossEntropyLoss(),
        num_categories=args.num_categories,
        device=device,
    )
    return SpectralSSM(configs)

def main():
    """
    Main function to run the benchmark.
    """
    torch.set_float32_matmul_precision("high")  # Enable CUDA TensorFloat-32

    parser = argparse.ArgumentParser(description="Benchmark script for Spectral SSM")
    parser.add_argument("--task", type=str, default="copy",
                        choices=["copy", "selective_copy", "adding", "induction", "associative"],
                        help="Benchmark task to run")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--num_categories", type=int, default=10, help="Number of categories (for applicable tasks)")
    parser.add_argument("--copy_len", type=int, default=10, help="Length of sequence to copy (for copy task)")
    parser.add_argument("--blank_len", type=int, default=5, help="Length of blank sequence (for copy task)")
    parser.add_argument("--sequence_len", type=int, default=30, help="Sequence length (for applicable tasks)")
    parser.add_argument("--vocab_size", type=int, default=20, help="Vocabulary size (for applicable tasks)")
    parser.add_argument("--selective", action="store_true", help="Use selective copy task")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for evaluation")

    # Model hyperparameters
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--n_embd", type=int, default=1, help="Embedding dimension")
    parser.add_argument("--d_in", type=int, default=1, help="Input dimension")
    parser.add_argument("--d_out", type=int, default=1, help="Output dimension")
    parser.add_argument("--d_proj", type=int, default=1, help="Projection dimension")
    parser.add_argument("--sl", type=int, default=300, help="Sequence length")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor")
    parser.add_argument("--bias", action="store_true", help="Use bias in the model")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--num_eigh", type=int, default=16, help="Number of eigenvalues")
    parser.add_argument("--k_y", type=int, default=2, help="k_y parameter")
    parser.add_argument("--k_u", type=int, default=3, help="k_u parameter")
    parser.add_argument("--learnable_m_y", action="store_true", help="Use learnable M_y")
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha parameter")
    parser.add_argument("--use_ar_y", action="store_true", help="Use AR-Y")
    parser.add_argument("--use_ar_u", action="store_true", help="Use AR-U")
    parser.add_argument("--use_hankel_L", action="store_true", help="Use Hankel-L")
    parser.add_argument("--moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts for MoE")
    parser.add_argument("--num_experts_per_timestep", type=int, default=2, help="Number of experts per timestep for MoE")

    parser.add_argument("--della", action="store_true", help="Running on Princeton Della cluster")
    args = parser.parse_args()

    device, local_rank, rank, world_size, main_process = setup(args)

    if main_process:
        colored_print("\nLyla: Hello! I'm here to assist with the benchmark tasks.", Colors.OKBLUE)

    # Generate dataset based on the chosen task
    if args.task == "copy":
        dataset = generate_copy(args.num_examples, args.num_categories, args.copy_len, args.blank_len, args.selective, args.seed)
    elif args.task == "adding":
        dataset = generate_adding(args.num_examples, args.sequence_len)
    elif args.task == "induction":
        dataset = generate_induction_heads(args.num_examples, args.sequence_len, args.vocab_size)
    elif args.task == "associative":
        dataset = generate_associative_recall(args.num_examples, args.sequence_len, args.vocab_size)

    # Create model
    model = get_model(args, device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Initialize benchmark
    benchmark = Benchmark(model, device)

    # Run benchmark
    datasets = [(args.task, dataset)]
    benchmark.benchmark(datasets, num_epochs=args.num_epochs, batch_size=args.batch_size)

    if main_process:
        colored_print("Lyla: Benchmark complete! I hope the results are insightful.", Colors.OKGREEN)

if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        cleanup()
