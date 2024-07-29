# =============================================================================#
# Authors: Windsor Nguyen, Isabel Liu
# File: benchmark.py
# =============================================================================#

"""Benchmarking on synthetic long-context datasets."""

import argparse
import time
import torch
from benchmarks.stu import SpectralSSM, SpectralSSMConfigs, ResidualSTU
from benchmarks.transformer import Transformer, TransformerConfigs
from benchmarks.hybrid import SpectralHybrid, SpectralHybridConfigs
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from torch.nn import CrossEntropyLoss, MSELoss
from utils.colors import Colors, colored_print
from utils.dist import setup, cleanup
from benchmarks.synthetic import (
    generate_copy,
    generate_adding,
    generate_mode_tracking,
    generate_induction_heads,
    generate_associative_recall,
    generate_multi_scale_adaptive,
    generate_needle_in_haystack,
    generate_telephone_book,
)


def get_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    sampler=None,
    distributed=False,
    pin_memory=False,
):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=True,
    )


def print_dataset_info(train_loader, world_size, local_rank, main_process):
    local_batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    local_size = num_batches * local_batch_size

    if world_size > 1:
        global_size = torch.tensor(local_size, dtype=torch.int64, device="cuda")
        torch.distributed.all_reduce(global_size, op=torch.distributed.ReduceOp.SUM)
        global_size = global_size.item()
    else:
        global_size = local_size

    if main_process:  # Only print on main process
        print(f"Global dataset size: {global_size}")
        print(f"Local dataset size on rank {local_rank}: {local_size}")
        print(f"Number of batches on rank {local_rank}: {num_batches}")
        print(f"Local batch size: {local_batch_size}")
        print(f"Effective global batch size: {world_size * local_batch_size}")


class Benchmark:
    """
    A class for benchmarking the Spectral SSM model on various tasks.
    """

    def __init__(self, model, task, max_grad_norm, device=None):
        """
        Initialize the Benchmark class.

        Args:
            model (torch.nn.Module): The model to benchmark.
            device (torch.device, optional): The device to run the model on.
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.lr = 1.8e-3
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            fused=(device.type == "cuda"),
        )
        self.task = task
        self.max_grad_norm = max_grad_norm
        self.time = None

        if self.task == "adding":
            self.criterion = MSELoss()
        elif self.task == "copy" or self.task == "induction":
            self.criterion = CrossEntropyLoss()
        else:
            raise ValueError("Task not yet supported")

    def train(self, dataloader, num_epochs=3):
        """
        Train the model for a specified number of epochs.

        Args:
            dataloader (DataLoader): DataLoader for the training dataset.
            num_epochs (int): Number of epochs to train. Default is 1.

        Returns:
            None
        """
        self.model.train()
        for current_epoch in range(num_epochs):
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(current_epoch)

            total_loss = 0
            num_batches = len(dataloader)
            for batch, (inputs, targets) in enumerate(
                tqdm(dataloader, desc=f"Training Epoch {current_epoch+1}/{num_epochs}")
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                _, loss = self.model(inputs, targets)

                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected: {loss.item()}. Skipping this batch.")
                    continue

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                total_loss += loss.item()

                # Print loss every 10 steps
                if (batch + 1) % 10 == 0:
                    print(
                        f"Epoch {current_epoch+1} | Batch {batch+1}/{num_batches} | Norm: {norm:.4f} | Loss: {loss.item():.4f}"
                    )

            epoch_loss = total_loss / num_batches
            print(
                f"Epoch {current_epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}"
            )

    def evaluate(self, dataset_name, dataloader):
        """
        Evaluate the model on a given dataset.

        Args:
            dataset_name (str): Name of the dataset.
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            tuple: A tuple containing:
                - avg_loss (float): The average loss over the dataset.
                - accuracy (float): The accuracy as a percentage.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch, (inputs, targets) in enumerate(
                tqdm(dataloader, desc=f"Evaluating on {dataset_name}")
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                preds, loss = self.model(inputs, targets)

                total_loss += loss.item()
                
                if self.task in ["copy", "induction"]:
                    total_correct += (preds == targets).sum().item()
                    total_samples += targets.numel()
                elif self.task == "adding":
                    total_samples += targets.numel()
                else:
                    colored_print(
                        f"Warning: Output shape {preds.shape} doesn't match target shape {targets.shape}. "
                        f"Skipping accuracy calculation for this batch.",
                        Colors.WARNING,
                    )

                # Print loss every 10 steps
                if (batch + 1) % 10 == 0:
                    print(
                        f"Evaluation Batch {batch+1}/{len(dataloader)}, Loss: {loss.item():.4f}"
                    )


        if torch.distributed.is_initialized():
            metrics = torch.tensor([total_loss, total_correct, total_samples]).to(self.device)
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            total_loss, total_correct, total_samples = metrics.tolist()

        avg_loss = total_loss / len(dataloader)
        
        if self.task == "adding":
            return avg_loss, avg_loss
        else:
            accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
            return avg_loss, accuracy

    def benchmark(self, train_loader, val_loader, num_epochs=3):
        """
        Run the benchmark on a dataset.

        Args:
            dataset_name (str): Name of the dataset.
            dataloader (DataLoader): DataLoader for the dataset.
            num_epochs (int): Number of epochs to train.
        """
        # Train
        start = time.time()
        self.train(train_loader, num_epochs=num_epochs)
        end = time.time()
        self.time = end - start

        # Evaluate
        loss, accuracy = self.evaluate(self.task, val_loader)

        colored_print(f"Dataset: {self.task}", Colors.HEADER)
        colored_print(f"  Validation Loss: {loss:.4f}", Colors.OKBLUE)
        
        if self.task in ["copy", "induction"]:
            colored_print(f"  Accuracy: {accuracy:.2f}%", Colors.OKGREEN)
        elif self.task in ["adding"]:
            colored_print(f"  Mean Squared Error: {loss:.4f}", Colors.OKGREEN)

        colored_print(f"  Training Time: {self.time:.2f} seconds", Colors.OKBLUE)
        print()


def get_input_dim(args):
    if args.task in [
        "copy",
        "selective_copy",
        "induction",
        "associative",
        "needle_in_haystack",
        "telephone_book",
    ]:
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
        # python -m benchmarks.benchmark --model transformer --learnable_m_y --task {task} --flash_attn --sub_rn
        configs = TransformerConfigs(
            # General Transformer settings
            n_layers=args.n_layers,
            n_embd=args.n_embd,
            n_heads=args.n_heads,
            d_in=get_input_dim(args),
            d_out=get_output_dim(args),
            sl=args.sl,
            ffn_scale=args.ffn_scale,
            embd_scale=args.embd_scale,
            sub_rn=args.sub_rn,
            bias=args.bias,
            dropout=args.dropout,
            flash_attn=args.flash_attn,
            use_sq_relu=args.use_sq_relu,
            task=args.task,
            vocab_size=args.vocab_size,
            loss_fn=MSELoss() if args.task == "adding" else CrossEntropyLoss(),
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
        # python -m benchmarks.benchmark --model stu --learnable_m_y --task {task} --use_ar_y --use_ar_u
        configs = SpectralSSMConfigs(
            n_layers=args.n_layers,
            n_embd=args.n_embd,
            d_in=get_input_dim(args),
            d_out=get_output_dim(args),
            sl=args.sequence_len,
            mlp_scale=args.mlp_scale,
            embd_scale=args.embd_scale,
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
            vocab_size=args.vocab_size,
            loss_fn=MSELoss() if args.task == "adding" else CrossEntropyLoss(),
            device=device,
        )
        model = SpectralSSM(configs).to(device)
        # model = ResidualSTU(configs, num_models=args.num_residual_models).to(device)

    if args.model == "hybrid":
        configs = SpectralHybridConfigs(
            n_layers=args.n_layers,
            n_embd=args.n_embd,
            n_heads=args.n_heads,
            d_in=get_input_dim(args),
            d_out=get_output_dim(args),
            sl=args.sequence_len,
            mlp_scale=args.mlp_scale,
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
            vocab_size=args.vocab_size,
            loss_fn=MSELoss() if args.task == "adding" else CrossEntropyLoss(),
            device=device,
            flash_attn=args.flash_attn,
            use_sq_relu=args.use_sq_relu,
            dilated_attn=args.dilated_attn,
            segment_lengths=args.segment_lengths,
            dilated_ratios=args.dilated_ratios,
            seq_parallel=args.seq_parallel,
            xpos_rel_pos=args.xpos_rel_pos,
            xpos_scale_base=args.xpos_scale_base,
            rms_norm_eps=args.rms_norm_eps,
            multiway=args.multiway,
        )
        model = SpectralHybrid(configs).to(device)

    return model


def main():
    """
    Main function to run the benchmark.
    """
    torch.set_float32_matmul_precision("high")  # Enable CUDA TensorFloat-32

    # TODO: Move these flags to a separate flags file or something.
    parser = argparse.ArgumentParser(description="Synthetic benchmark script")
    parser.add_argument(
        "--task",
        type=str,
        default="copy",
        choices=[
            "copy",
            "adding",
            "mode",
            "induction",
            "associative",
            "multi_scale_adaptive",
            "needle_in_haystack",
            "telephone_book",
        ],
        help="Benchmark task to run",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="stu",
        choices=["stu", "transformer", "hybrid", "mamba"],
        help="Model to benchmark",
    )
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm"
    )
    parser.add_argument(
        "--num_examples", type=int, default=512, help="Number of examples to generate"
    )
    parser.add_argument(
        "--num_categories",
        type=int,
        default=10,
        help="Number of categories (for applicable tasks)",
    )
    parser.add_argument(
        "--copy_len",
        type=int,
        default=10,
        help="Length of sequence to copy (for copy task)",
    )
    parser.add_argument(
        "--blank_len",
        type=int,
        default=5,
        help="Length of blank sequence (for copy task)",
    )
    parser.add_argument(
        "--sequence_len",
        type=int,
        default=30,
        help="Sequence length (for applicable tasks)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=20,
        help="Vocabulary size (for applicable tasks)",
    )
    parser.add_argument(
        "--selective", action="store_true", help="Use selective copy task"
    )
    parser.add_argument(
        "--p", type=int, default=None, help="Modulo operation for the adding task"
    )
    # For multi_scale_adaptive task
    parser.add_argument(
        "--num_regimes",
        type=int,
        default=3,
        help="Number of different LDS regimes (for multi_scale_adaptive task)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.1,
        help="Noise level (for multi_scale_adaptive task)",
    )

    # For needle_in_haystack task
    parser.add_argument(
        "--needle_len",
        type=int,
        default=5,
        help="Length of the needle sequence (for needle_in_haystack task)",
    )

    # For telephone_book task
    parser.add_argument(
        "--num_entries",
        type=int,
        default=100,
        help="Number of entries in the telephone book (for telephone_book task)",
    )
    parser.add_argument(
        "--name_len",
        type=int,
        default=10,
        help="Length of each name in the telephone book (for telephone_book task)",
    )
    parser.add_argument(
        "--number_len",
        type=int,
        default=10,
        help="Length of each number in the telephone book (for telephone_book task)",
    )

    parser.add_argument("--seed", type=int, default=1_337, help="Random seed")
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for evaluation"
    )

    # Model hyperparameters
    parser.add_argument(
        "--n_layers", type=int, default=1, help="Number of layers in the model"
    )
    parser.add_argument("--d_model", type=int, default=32, help="Model dimension")
    parser.add_argument("--n_embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--n_heads", type=int, default=1, help="Number of attention heads"
    )
    parser.add_argument("--d_in", type=int, default=1, help="Input dimension")
    parser.add_argument("--d_out", type=int, default=1, help="Output dimension")
    parser.add_argument("--sl", type=int, default=1000, help="Sequence length")
    parser.add_argument("--mlp_scale", type=float, default=4, help="MLP scale factor")
    parser.add_argument("--ffn_scale", type=int, default=4, help="FFN scale factor")
    parser.add_argument("--embd_scale", type=int, default=4, help="Embedding scale factor")
    parser.add_argument("--sub_rn", action="store_true", help="Use sub-layer RMS Norm")
    parser.add_argument("--bias", action="store_true", help="Use bias in the model")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--pct_attn", type=float, default=0.5, help="Percentage of layers using attention")
    parser.add_argument("--flash_attn", action="store_true", help="Use FlashAttention")
    parser.add_argument("--use_sq_relu", action="store_true", help="Use Squared ReLU")
    parser.add_argument(
        "--num_eigh", type=int, default=16, help="Number of eigenvalues"
    )
    parser.add_argument("--k_y", type=int, default=2, help="k_y parameter")
    parser.add_argument("--k_u", type=int, default=3, help="k_u parameter")
    parser.add_argument(
        "--learnable_m_y", action="store_true", help="Use learnable M_y"
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha parameter")
    parser.add_argument("--use_ar_y", action="store_true", help="Use AR-Y")
    parser.add_argument("--use_ar_u", action="store_true", help="Use AR-U")
    parser.add_argument("--use_hankel_L", action="store_true", help="Use Hankel-L")
    parser.add_argument("--moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument(
        "--num_experts", type=int, default=3, help="Number of experts for MoE"
    )
    parser.add_argument(
        "--num_experts_per_timestep",
        type=int,
        default=2,
        help="Number of experts per timestep for MoE",
    )
    parser.add_argument(
        "--num_residual_models",
        type=int,
        default=3,
        help="Number of residual STU models",
    )

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

    parser.add_argument(
        "--della", action="store_true", help="Running on Princeton Della cluster"
    )
    args = parser.parse_args()

    device, local_rank, rank, world_size, main_process = setup(args)

    if main_process:
        colored_print(
            "\nLyla: Hello! I'm here to assist with the benchmark tasks.", Colors.OKBLUE
        )

    # Generate dataset based on the chosen task
    if args.task == "copy":
        dataset = generate_copy(
            args.num_examples,
            args.num_categories,
            args.copy_len,
            args.blank_len,
            args.selective,
            args.seed,
        )
    elif args.task == "adding":
        dataset = (
            generate_adding(args.num_examples, args.sequence_len, args.seed)
            if args.p is None
            else generate_adding(
                args.num_examples, args.sequence_len, args.p, args.seed
            )
        )
    elif args.task == "mode":
        dataset = generate_mode_tracking(
            args.num_examples, args.sequence_len, args.num_categories, args.seed
        )
    elif args.task == "induction":
        dataset = generate_induction_heads(
            args.num_examples, args.sequence_len, args.vocab_size, args.seed
        )
    elif args.task == "associative":
        dataset = generate_associative_recall(
            args.num_examples, args.sequence_len, args.vocab_size, args.seed
        )
    elif args.task == "multi_scale_adaptive":
        dataset = generate_multi_scale_adaptive(
            args.num_examples,
            args.sequence_len,
            args.num_regimes,
            args.noise_level,
            args.seed,
        )
    elif args.task == "needle_in_haystack":
        dataset = generate_needle_in_haystack(
            args.num_examples,
            args.sequence_len,
            args.needle_len,
            args.vocab_size,
            args.seed,
        )
    elif args.task == "telephone_book":
        dataset = generate_telephone_book(
            args.num_examples,
            args.num_entries,
            args.name_len,
            args.number_len,
            args.vocab_size,
            args.seed,
        )
    assert args.batch_size <= len(
        dataset
    ), "Batch size must be less than the dataset size."

    indices = torch.randperm(len(dataset))
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset, indices[train_size:])

    train_sampler = (
        DistributedSampler(
            dataset=train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
        if world_size > 1
        else None
    )

    val_sampler = (
        DistributedSampler(
            dataset=val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
        )
        if world_size > 1
        else None
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        distributed=(world_size > 1),
        pin_memory=(device == torch.device("cuda")),
    )
    print_dataset_info(train_loader, world_size, local_rank, main_process)

    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        distributed=(world_size > 1),
        pin_memory=(device == torch.device("cuda")),
    )

    # Create model
    model = get_model(args, device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    model = model.module if world_size > 1 else model
    if args.compile:
        model = torch.compile(model)

    # Initialize benchmark
    benchmark = Benchmark(model, args.task, args.max_grad_norm, device)

    # Run benchmark
    benchmark.benchmark(train_loader, val_loader, num_epochs=args.num_epochs)

    if main_process:
        colored_print(
            "Lyla: Benchmark complete! I hope the results are insightful.",
            Colors.OKGREEN,
        )
        print(f"Length of dataset: {len(dataset)}")
        colored_print(
            f"Total time steps: {len(train_dataset) * args.num_epochs}",
            Colors.OKGREEN,
        )
        colored_print(
            f"Time taken for training: {benchmark.time:.2f} seconds", Colors.WARNING
        )

        # Print out the configs used by the specific generator function
        colored_print("\nConfigs used for data generation:", Colors.HEADER)
        if args.task == "copy":
            print("Task: Copy")
            print(f"Number of examples: {args.num_examples}")
            print(f"Number of categories: {args.num_categories}")
            print(f"Copy length: {args.copy_len}")
            print(f"Blank length: {args.blank_len}")
            print(f"Selective: {args.selective}")
            print(f"Seed: {args.seed}")
        elif args.task == "adding":
            print("Task: Adding")
            print(f"Number of examples: {args.num_examples}")
            print(f"Sequence length: {args.sequence_len}")
            print(f"Seed: {args.seed}")
            if args.p is not None:
                print(f"Modulo: {args.p}")
        elif args.task == "mode":
            raise ValueError("Mode tracking task not yet supported")
        elif args.task == "induction":
            print("Task: Induction Heads")
            print(f"Number of examples: {args.num_examples}")
            print(f"Sequence length: {args.sequence_len}")
            print(f"Vocabulary size: {args.vocab_size}")
            print(f"Seed: {args.seed}")
        elif args.task == "associative":
            print("Task: Associative Recall")
            print(f"Number of examples: {args.num_examples}")
            print(f"Sequence length: {args.sequence_len}")
            print(f"Vocabulary size: {args.vocab_size}")
            print(f"Seed: {args.seed}")
        elif args.task == "multi_scale_adaptive":
            print("Task: Multi-scale Adaptive")
            print(f"Number of examples: {args.num_examples}")
            print(f"Sequence length: {args.sequence_len}")
            print(f"Number of regimes: {args.num_regimes}")
            print(f"Noise level: {args.noise_level}")
            print(f"Seed: {args.seed}")
        elif args.task == "needle_in_haystack":
            print("Task: Needle in Haystack")
            print(f"Number of examples: {args.num_examples}")
            print(f"Sequence length: {args.sequence_len}")
            print(f"Needle length: {args.needle_len}")
            print(f"Vocabulary size: {args.vocab_size}")
            print(f"Seed: {args.seed}")
        elif args.task == "telephone_book":
            print("Task: Telephone Book")
            print(f"Number of examples: {args.num_examples}")
            print(f"Number of entries: {args.num_entries}")
            print(f"Name length: {args.name_len}")
            print(f"Number length: {args.number_len}")
            print(f"Vocabulary size: {args.vocab_size}")
            print(f"Seed: {args.seed}\n")


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        cleanup()
