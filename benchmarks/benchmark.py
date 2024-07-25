# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: benchmark.py
# =============================================================================#

"""Benchmarking on synthetic long-context datasets."""

import argparse
import torch
from benchmarks.stu import SpectralSSM, SpectralSSMConfigs, ResidualSTU
from benchmarks.transformer import Transformer, TransformerConfigs
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from utils.colors import Colors, colored_print
from utils.dist import setup, cleanup
from benchmarks.synthetic import (
    generate_copy,
    generate_adding,
    generate_induction_heads,
    generate_associative_recall,
    generate_multi_scale_adaptive,
    generate_needle_in_haystack,
    generate_telephone_book,
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
        self.godly_lr = 5e-3
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.godly_lr)

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

def get_input_dim(args):
    if args.task in ["copy", "selective_copy", "induction", "associative", "needle_in_haystack", "telephone_book"]:
        return args.vocab_size
    elif args.task == "adding":
        return 2  # The adding task has 2 input channels
    elif args.task == "multi_scale_adaptive":
        return 2  # The multi_scale_adaptive task has 2-dimensional state
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
def get_output_dim(args):
    if args.task in ["copy", "selective_copy", "induction", "associative"]:
        return args.vocab_size
    elif args.task == "adding":
        return 1  # The adding task has a single output (the sum)
    elif args.task == "multi_scale_adaptive":
        return 2  # The multi_scale_adaptive task predicts the next 2D state
    elif args.task == "needle_in_haystack":
        return args.sequence_len  # Predicting the position of the needle
    elif args.task == "telephone_book":
        return 10  # Assuming phone numbers are digits 0-9
    else:
        raise ValueError(f"Unknown task: {args.task}")

def get_model(args, device):
    """
    Create and configure the Spectral SSM model.

    Args:
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device to run the model on.

    Returns:
        SpectralSSM: Configured Spectral SSM model.
    """
    if args.model == "transformer":
        configs = TransformerConfigs(
        # General Transformer settings
        n_layers=args.n_layers,
        n_embd=args.n_embd,
        n_heads=args.n_heads,
        d_in=get_input_dim(args),
        d_out=get_output_dim(args),
        sl=args.sl,
        scale=args.scale,
        sub_rn=args.sub_rn,
        bias=args.bias,
        dropout=args.dropout,
        flash_attn=args.flash_attn,
        use_sq_relu=args.use_sq_relu,
        task=args.task,
        loss_fn=CrossEntropyLoss(),
        device=device,

        # MoE
        moe=args.moe,
        num_experts=args.num_experts,
        num_experts_per_timestep=args.num_experts_per_timestep,

        # Dilated Attention
        dilated_attn=args.dilated_attn,
        segment_lengths=args.segment_lengths,
        dilated_ratios=args.dilated_ratios,
        seq_parallel=args.seq_parallel,
        xpos_rel_pos=args.xpos_rel_pos,
        xpos_scale_base=args.xpos_scale_base,
        rms_norm_eps=args.rms_norm_eps,
        multiway=args.multiway,
    )
        model = Transformer(configs).to(device)

    if args.model == "stu":
        configs = SpectralSSMConfigs(
            n_layers=args.n_layers,
            n_embd=args.n_embd,
            d_in=args.num_categories if args.task == "copy" else args.sequence_len,
            d_out=args.num_categories if args.task == "copy" else args.sequence_len,
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
            task=args.task,
            loss_fn=CrossEntropyLoss(),
            device=device,
        )
        model = SpectralSSM(configs).to(device)
        # model = ResidualSTU(configs, num_models=args.num_residual_models).to(device)
        
    return model

def main():
    """
    Main function to run the benchmark.
    """
    torch.set_float32_matmul_precision("high")  # Enable CUDA TensorFloat-32

    parser = argparse.ArgumentParser(description="Synthetic benchmark script")
    parser.add_argument("--task", type=str, default="copy",
                        choices=["copy", "adding", "induction", "associative", 
                                 "multi_scale_adaptive", "needle_in_haystack", 
                                 "telephone_book"],
                        help="Benchmark task to run")

    parser.add_argument("--model", type=str, default="stu",
                        choices=["stu", "transformer", "hybrid", "mamba"],
                        help="Model to benchmark")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to generate")
    parser.add_argument("--num_categories", type=int, default=10, help="Number of categories (for applicable tasks)")
    parser.add_argument("--copy_len", type=int, default=10, help="Length of sequence to copy (for copy task)")
    parser.add_argument("--blank_len", type=int, default=5, help="Length of blank sequence (for copy task)")
    parser.add_argument("--sequence_len", type=int, default=30, help="Sequence length (for applicable tasks)")
    parser.add_argument("--vocab_size", type=int, default=20, help="Vocabulary size (for applicable tasks)")
    parser.add_argument("--selective", action="store_true", help="Use selective copy task")
    # For multi_scale_adaptive task
    parser.add_argument("--num_regimes", type=int, default=3, help="Number of different LDS regimes (for multi_scale_adaptive task)")
    parser.add_argument("--noise_level", type=float, default=0.1, help="Noise level (for multi_scale_adaptive task)")

    # For needle_in_haystack task
    parser.add_argument("--needle_len", type=int, default=5, help="Length of the needle sequence (for needle_in_haystack task)")

    # For telephone_book task
    parser.add_argument("--num_entries", type=int, default=100, help="Number of entries in the telephone book (for telephone_book task)")
    parser.add_argument("--name_len", type=int, default=10, help="Length of each name in the telephone book (for telephone_book task)")
    parser.add_argument("--number_len", type=int, default=10, help="Length of each number in the telephone book (for telephone_book task)")

    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for evaluation")

    # Model hyperparameters
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--n_embd", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--d_in", type=int, default=1, help="Input dimension")
    parser.add_argument("--d_out", type=int, default=1, help="Output dimension")
    parser.add_argument("--sl", type=int, default=300, help="Sequence length")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor")
    parser.add_argument("--sub_rn", action="store_true", help="Use sub-layer RMS Norm")
    parser.add_argument("--bias", action="store_true", help="Use bias in the model")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--flash_attn", action="store_true", help="Use FlashAttention")
    parser.add_argument("--use_sq_relu", action="store_true", help="Use Squared ReLU")
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
    parser.add_argument("--num_residual_models", type=int, default=3, help="Number of residual STU models")

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
        default=False,
        help="Whether to use relative positional embeddings. Defaults to False.",
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

    parser.add_argument("--della", action="store_true", help="Running on Princeton Della cluster")
    args = parser.parse_args()

    device, local_rank, rank, world_size, main_process = setup(args)

    if main_process:
        colored_print("\nLyla: Hello! I'm here to assist with the benchmark tasks.", Colors.OKBLUE)

    # Generate dataset based on the chosen task
    if args.task == "copy":
        dataset = generate_copy(args.num_examples, args.num_categories, args.copy_len, args.blank_len, args.selective, args.seed)
    elif args.task == "adding":
        dataset = generate_adding(args.num_examples, args.sequence_len, args.seed)
    elif args.task == "induction":
        dataset = generate_induction_heads(args.num_examples, args.sequence_len, args.vocab_size, args.seed)
    elif args.task == "associative":
        dataset = generate_associative_recall(args.num_examples, args.sequence_len, args.vocab_size, args.seed)
    elif args.task == "multi_scale_adaptive":
        dataset = generate_multi_scale_adaptive(args.num_examples, args.sequence_len, args.num_regimes, args.noise_level, args.seed)
    elif args.task == "needle_in_haystack":
        dataset = generate_needle_in_haystack(args.num_examples, args.sequence_len, args.needle_len, args.vocab_size, args.seed)
    elif args.task == "telephone_book":
        dataset = generate_telephone_book(args.num_examples, args.num_entries, args.name_len, args.number_len, args.vocab_size, args.seed)

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
