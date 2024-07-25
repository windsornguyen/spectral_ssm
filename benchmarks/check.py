import numpy as np
import torch
from torch.utils.data import TensorDataset

def generate_associative_recall_torch(
    num_examples: int = 5,
    sequence_len: int = 30,
    vocab_size: int = 10,
    seed: int = 0,
) -> TensorDataset:
    """Generate an associative recall task.

    This associative recall task is taken from (Dao, Fu, Saab, 2023). From their
    paper:

      Associative Recall is similar to the induction head task, but requires the
      model to remember multiple key-value pairs. At the end of the sequence, the
      model must recall a specific value belonging to a specific key.

    Questions:
      - Should each example have a new association?
      - Should we use the same tokens each time?

    Args:
      num_examples: Number of examples to generate.
      sequence_len: Length of each sequence (adjusted to be even).
      vocab_size: Size of the vocabulary.
      seed: Seed for random number generator.

    Returns:
      A PyTorch dataset.
    """
    # # Set random seed.
    # torch.manual_seed(seed)
    rng = np.random.default_rng(seed=seed)
    idx = torch.from_numpy(rng.choice(vocab_size, (num_examples, sequence_len // 2), replace=True))

    def get_assoc(start: int, end: int):
        # Range of values
        x = torch.arange(start, end)

        # Make num_examples copies.
        x = x.repeat(num_examples, 1)

        # # Shuffle each row independently.
        x = x[torch.randperm(x.size(1))]
        # x = x[torch.from_numpy(rng.permutation(num_examples))]

        # Grab the corresponding indices
        return x[torch.arange(num_examples).unsqueeze(1), idx]

    keys = get_assoc(0, vocab_size)
    vals = get_assoc(vocab_size, 2 * vocab_size)

    # Interleave keys and values by row.
    inputs = torch.zeros((num_examples, sequence_len), dtype=keys.dtype)
    inputs[:, 0::2] = keys
    inputs[:, 1::2] = vals

    # Get key we want to find associated value for.
    # query_idx = torch.randint(0, vocab_size, (num_examples,))
    query_idx = torch.from_numpy(rng.choice(vocab_size, num_examples, replace=True))
    query_keys = keys[torch.arange(num_examples), query_idx].unsqueeze(1)
    inputs = torch.cat((inputs, query_keys), dim=1)
    outputs = vals[torch.arange(num_examples), query_idx]

    return inputs, outputs


def generate_associative_recall_numpy(
    num_examples: int = 5,
    sequence_len: int = 30,
    vocab_size: int = 10,
    seed: int = 0,
) -> TensorDataset:
    """Generate an associative recall task.

    This associative recall task is taken from (Dao, Fu, Saab, 2023). From their
    paper:

    Associative Recall is similar to the induction head task, but requires the
    model to remember multiple key-value pairs. At the end of the sequence, the
    model must recall a specific value belonging to a specific key.

    Questions:
    - Should each example have a new association?
    - Should we use the same tokens each time?

    Args:
    num_examples: Number of examples to generate.
    sequence_len: Length of each sequence.
    vocab_size: Size of the vocabulary.
    seed: Seed for random number generator.

    Returns:
    A Tensorflow dataset.
    """
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(vocab_size, (num_examples, sequence_len // 2), replace=True)

    def get_assoc(start: int, end: int):
        # Range of values
        x = np.arange(start, end)

        # Make num_examples copies.
        x = np.tile(x, (num_examples, 1))

        # Shuffle each row independently.
        x = rng.permuted(x, axis=1)

        # Grab the corresponding indices
        return np.take(x, idx)

    keys = get_assoc(0, vocab_size)
    vals = get_assoc(vocab_size, 2 * vocab_size)

    # Interleave keys and values by row.
    inputs = np.zeros((num_examples, sequence_len), dtype=keys.dtype)
    inputs[:, 0::2] = keys
    inputs[:, 1::2] = vals

    # Get key we want to find associated value for.
    idx = rng.choice(vocab_size, num_examples, replace=True)
    keys = np.expand_dims(keys[np.arange(num_examples), idx], axis=1)
    inputs = np.hstack((inputs, keys))
    outputs = vals[np.arange(num_examples), idx].squeeze()

    return inputs, outputs

# Comparison function
def compare_outputs(
    num_examples: int = 5,
    sequence_len: int = 30,
    vocab_size: int = 10,
    seed: int = 0,
):
    np_inputs, np_outputs = generate_associative_recall_numpy(num_examples, sequence_len, vocab_size, seed)
    torch_inputs, torch_outputs = generate_associative_recall_torch(num_examples, sequence_len, vocab_size, seed)

    inputs_match = np.array_equal(np_inputs, torch_inputs.numpy())
    outputs_match = np.array_equal(np_outputs, torch_outputs.numpy())

    print(f"Inputs match: {inputs_match}")
    print(f"Outputs match: {outputs_match}")

    if not inputs_match:
        print("Input differences:")
        print("NumPy inputs:")
        print(np_inputs)
        print("PyTorch inputs:")
        print(torch_inputs.numpy())

    if not outputs_match:
        print("Output differences:")
        print("NumPy outputs:")
        print(np_outputs)
        print("PyTorch outputs:")
        print(torch_outputs.numpy())

# Run comparison
print("Comparing outputs:")
compare_outputs(num_examples=5, sequence_len=30, vocab_size=10, seed=42)
