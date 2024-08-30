"""
Synthetic long-context benchmarks.
"""

import numpy as np
import random
import torch
from torch.utils.data import TensorDataset


def generate_copy(
    num_examples: int = 10,
    num_categories: int = 10,
    copy_len: int = 10,
    blank_len: int = 5,
    selective: bool = False,
    seed: int = 1_337,
) -> TensorDataset:
    """
    Generate a copy task dataset based on Arjovsky, Shah, and Bengio (2016).

    Task Description:
    - Input sequence: [copy_sequence][blank_tokens][delimiter][blank_tokens]
    - Output sequence: [blank_tokens][copy_sequence]

    The task requires remembering a categorical sequence for a variable number of time steps.

    Args:
        num_examples: Number of examples to generate.
        num_categories: Number of token categories.
            - Categories 0 to num_categories-3: Tokens to be copied
            - Category num_categories-2: Blank token
            - Category num_categories-1: Delimiter token
        copy_len: Length of the sequence to be copied.
        blank_len: Number of blank tokens between copy and paste.
        selective: If True, shuffles blank spaces into the sequence to be copied.
        seed: Random seed for reproducibility.

    Returns:
        TensorDataset with:
            - inputs: Shape (num_examples, 2*copy_len + blank_len)
            - targets: Shape (num_examples, blank_len + copy_len)

    Example:
        >>> dataset = generate_copy(num_examples=100, copy_len=8, blank_len=3)
        >>> inputs, targets = dataset[0]
        >>> print(inputs.shape, targets.shape)
        torch.Size([19]) torch.Size([11])

    Note:
        A memoryless baseline strategy would predict the blank token for the first
        (blank_len + copy_len) steps, then random tokens, yielding a categorical
        cross entropy of (copy_len * log(num_categories-2)) / (2*copy_len + blank_len).
    """
    # Assign characters
    copy_chars = torch.arange(num_categories - 2)
    blank_char = num_categories - 2
    delim_char = num_categories - 1

    # Set random seed
    torch.manual_seed(seed)

    # Construct input sequences
    to_copy = torch.randint(0, len(copy_chars), (num_examples, copy_len))
    blank = torch.full((num_examples, blank_len - 1), blank_char)
    delim = torch.full((num_examples, 1), delim_char)
    to_fill = torch.full((num_examples, copy_len), blank_char)

    if selective:

        def insert_blanks(row):
            insert_positions = torch.randperm(copy_len)[: blank_len - 1]
            inserted = torch.full((copy_len + blank_len - 1,), blank_char)
            mask = torch.ones(copy_len + blank_len - 1, dtype=torch.bool)
            mask[insert_positions] = False
            inserted[mask] = row
            return inserted

        inputs = torch.stack([insert_blanks(row) for row in to_copy])
    else:
        inputs = torch.cat((to_copy, blank), dim=1)

    inputs = torch.cat((inputs, delim, to_fill), dim=1)

    # Construct output sequences
    blank_output = torch.full((num_examples, blank_len + copy_len), blank_char)
    outputs = torch.cat((blank_output, to_copy), dim=1)
    return TensorDataset(inputs, outputs)


def generate_adding(
    num_examples: int = 10,
    sequence_len: int = 128,
    p: int = None,
    seed: int = 1_337,
) -> TensorDataset:
    """
    Generate an adding task with an optional modulo operation.

    This adding task is adapted from (Arjovsky, Shah, and Bengio, 2016). From their
    paper:

      We closely follow the adding problem defined in (Hochreiter & Schmidhuber,
      1997) to explain the task at hand. Each input consists of two sequences of
      length T. The first sequence, which we denote x, consists of numbers sampled
      uniformly at random U[0, 1]. The second sequence is an indicator sequence
      consisting of exactly two entries of 1 and remaining entries 0. The first 1
      entry is located uniformly at random in the first half of the sequence,
      whilst the second 1 entry is located uniformly at random in the second half.
      The output is the sum of the two entries of the first sequence,
      corresponding to where the 1 entries are located in the second sequence.

      IMPORTANT: A naive strategy of predicting 1 as the output regardless of the
      input sequence gives an expected mean squared error of 0.167, the variance
      of the sum of two independent uniform distributions. This is our baseline to beat.

    Args:
      num_examples: Number of examples to generate.
      sequence_len: Length of each sequence.
      p: If provided, the sum will be computed modulo p. If None, no modulo
         operation is performed.
      seed: Random seed for reproducibility.

    Returns:
      A PyTorch dataset.
    """
    # Set random seed.
    torch.manual_seed(seed)

    # Construct the first sequence.
    seq_1 = torch.rand((num_examples, sequence_len))

    # Construct the second sequence.
    seq_2 = torch.zeros((num_examples, sequence_len))
    idx_1 = torch.randint(0, sequence_len // 2, (num_examples,))
    idx_2 = torch.randint(sequence_len // 2, sequence_len, (num_examples,))
    seq_2[torch.arange(num_examples), idx_1] = 1
    seq_2[torch.arange(num_examples), idx_2] = 1

    # Compute the outputs.
    outputs = torch.sum(seq_1 * seq_2, dim=1)

    # Apply modulo operation if p is provided
    if p is not None:
        outputs = outputs % p

    # Concatenate the inputs.
    inputs = torch.cat((seq_1, seq_2), dim=1)

    # Construct dataset.
    return TensorDataset(inputs, outputs)


def generate_mode_tracking(
    num_examples: int = 10,
    sequence_len: int = 128,
    num_classes: int = 5,
    seed: int = 1_337,
) -> TensorDataset:
    """
    Generate a mode tracking task.

    In this task, each input consists of a sequence of integers. The output
    at each step is the mode (most frequent element) of all elements seen
    so far in the sequence. If there's a tie, the smallest number is chosen.

    Args:
      num_examples: Number of examples to generate.
      sequence_len: Length of each sequence.
      num_classes: Number of possible integer classes (0 to num_classes-1).
      seed: Random seed for reproducibility.

    Returns:
      A PyTorch dataset.
    """
    # Set random seed
    torch.manual_seed(seed)

    # Generate input sequences
    inputs = torch.randint(0, num_classes, (num_examples, sequence_len))

    # Compute outputs
    outputs = torch.zeros_like(inputs)
    for i in range(num_examples):
        counts = torch.zeros(num_classes)
        for j in range(sequence_len):
            counts[inputs[i, j]] += 1
            outputs[i, j] = torch.argmax(counts)

    # Construct dataset
    return TensorDataset(inputs, outputs)


def generate_induction_heads(
    num_examples: int = 10_000,
    sequence_len: int = 512,
    vocab_size: int = 20,
    min_prefix_len: int = 50,
    max_prefix_len: int = 100,
    min_pattern_len: int = 3,
    max_pattern_len: int = 4,
    num_patterns: int = 8,
    seed: int = 1_337,
    num_pairs: int = 5,
    assoc_recall: bool = False,
) -> TensorDataset:
    """
    Generates an induction heads task or an associative recall task.

    Args:
        num_examples: Number of sequences to generate.
        sequence_len: Length of each sequence.
        vocab_size: Size of the vocabulary (excluding special tokens).
        min_prefix_len: Minimum length of the prefix (for induction heads).
        max_prefix_len: Maximum length of the prefix (for induction heads).
        min_pattern_len: Minimum length of the pattern (for induction heads).
        max_pattern_len: Maximum length of the pattern (for induction heads).
        num_patterns: Number of patterns in each sequence (for induction heads).
        seed: Random seed for reproducibility.
        assoc_recall: If True, generate an associative recall task instead.
        num_pairs: Number of key-value pairs (for associative recall).

    Note:
        The CrossEntropyLoss objective function should set ignore_index=-1.

    Returns:
        A PyTorch dataset with input sequences and corresponding targets.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Special tokens
    START, END, PAD = vocab_size, vocab_size + 1, vocab_size + 2

    inputs = torch.full((num_examples, sequence_len), PAD, dtype=torch.long)
    targets = torch.full((num_examples, sequence_len), -1, dtype=torch.long)

    if assoc_recall:
        assert (
            sequence_len >= 2 * num_pairs + 1
        ), "Sequence length must be at least 2 * num_pairs + 1"

        for i in range(num_examples):
            inputs[i, 0] = START  # Start token
            idx = 1

            # Generate key-value pairs
            pairs = [
                (random.randint(0, vocab_size - 1), random.randint(0, vocab_size - 1))
                for _ in range(num_pairs)
            ]

            # Place key-value pairs
            for key, value in pairs:
                inputs[i, idx] = key
                inputs[i, idx + 1] = value
                idx += 2

            # Choose a random key to recall
            recall_key = random.choice([pair[0] for pair in pairs])
            inputs[i, idx] = recall_key

            # Set the target as the corresponding value
            target_value = next(pair[1] for pair in pairs if pair[0] == recall_key)
            targets[i, idx] = target_value

            inputs[i, -1] = END  # End token

    else:
        for i in range(num_examples):
            inputs[i, 0] = START  # Start token
            idx = 1

            # Generate patterns with variable-length prefixes
            patterns = []
            for _ in range(num_patterns):
                prefix_len = random.randint(min_prefix_len, max_prefix_len)
                pattern_len = random.randint(min_pattern_len, max_pattern_len)
                prefix = [random.randint(0, vocab_size - 1) for _ in range(prefix_len)]
                pattern = [
                    random.randint(0, vocab_size - 1) for _ in range(pattern_len)
                ]
                patterns.append((prefix, pattern))

            for prefix, pattern in patterns:
                total_len = len(prefix) + len(pattern)
                # Ensure space for pattern and gap
                if idx + total_len * 2 + 1 > sequence_len - 1:
                    break

                # First occurrence
                inputs[i, idx : idx + len(prefix)] = torch.tensor(
                    prefix, dtype=torch.long
                )
                inputs[i, idx + len(prefix) : idx + total_len] = torch.tensor(
                    pattern, dtype=torch.long
                )
                idx += total_len

                # Random gap (at least 1 token)
                remaining_space = sequence_len - idx - total_len - 1
                max_gap = max(
                    1,
                    remaining_space
                    // (num_patterns - patterns.index((prefix, pattern))),
                )
                gap = random.randint(1, max_gap)
                idx += gap

                # Second occurrence
                inputs[i, idx : idx + len(prefix)] = torch.tensor(
                    prefix, dtype=torch.long
                )
                inputs[i, idx + len(prefix) : idx + total_len] = torch.tensor(
                    pattern, dtype=torch.long
                )
                targets[i, idx + len(prefix) : idx + total_len] = torch.tensor(
                    pattern, dtype=torch.long
                )  # Set target, excluding prefix
                idx += total_len

            # Fill remaining space with random tokens
            while idx < sequence_len - 1:
                inputs[i, idx] = random.randint(0, vocab_size - 1)
                idx += 1

            inputs[i, -1] = END  # End token

    return TensorDataset(inputs, targets)


def generate_multi_scale_adaptive(
    num_examples: int = 10,
    sequence_len: int = 1000,
    num_regimes: int = 3,
    noise_level: float = 0.1,
    seed: int = 1_337,
) -> TensorDataset:
    """
    Generates a multi-scale adaptive learning task with switching linear
    dynamical systems.

    This task tests the model's ability to recognize patterns at different scales
    and adapt to changing rules. The sequence switches between different linear
    dynamical systems, with added noise.

    Args:
        num_examples: Number of examples to generate.
        sequence_len: Length of each sequence.
        num_regimes: Number of different LDS regimes to switch between.
        noise_level: Standard deviation of the Gaussian noise to add.
        seed: Random seed for reproducibility.

    Returns:
        A PyTorch dataset with inputs and targets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    A_matrices = [np.random.randn(2, 2) for _ in range(num_regimes)]

    inputs = []
    targets = []

    for _ in range(num_examples):
        x = np.zeros((sequence_len, 2))
        x[0] = np.random.randn(2)

        # Randomly assign regime changes
        regime_changes = np.sort(
            np.random.choice(sequence_len, num_regimes - 1, replace=False)
        )
        current_regime = 0

        for t in range(1, sequence_len):
            if t in regime_changes:
                current_regime += 1
            x[t] = A_matrices[current_regime] @ x[t - 1] + np.random.normal(
                0, noise_level, 2
            )

        inputs.append(x[:-1])
        targets.append(x[1:])

    inputs = torch.tensor(np.array(inputs, dtype=np.float32))
    targets = torch.tensor(np.array(targets, dtype=np.float32))

    return TensorDataset(inputs, targets)
