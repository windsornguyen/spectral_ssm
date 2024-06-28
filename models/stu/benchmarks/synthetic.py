# =============================================================================#
# Authors: Windsor Nguyen
# File: synthetic.py
# =============================================================================#

"""Synthetic long-context datasets."""

import torch
from torch.utils.data import TensorDataset

def generate_copy(
    num_examples: int = 5,
    num_categories: int = 10,
    copy_len: int = 10,
    blank_len: int = 5,
    selective: bool = False,
    seed: int = 0,
) -> TensorDataset:
    r"""Generate a copy task.

    This copy task is taken from (Arjovsky, Shah, and Bengio, 2016). From their
    paper:

      Following a similar setup to (Hochreiter & Schmidhuber, 1997), we outline
      the copy memory task. Consider 10 categories, a_0 to a_9. The input takes
      the form of a T + 20 length vector of categories, where we test over a range
      of values of T. The first 10 entries are sampled uniformly, independently
      and with replacement from a_0 to a_7, and represent the sequence which will
      need to be remembered. The next T − 1 entries are set to a_8, which can be
      thought of as the ’blank’ category. The next single entry is a_9, which
      represents a delimiter, which should indicate to the algorithm that it is
      now required to reproduce the initial 10 categories in the output. The
      remaining 10 entries are set to a_8. The required output sequence consists
      of T + 10 repeated entries of a_8, followed by the first 10 categories of
      the input sequence in exactly the same order. The goal is to minimize the
      average cross entropy of category predictions at each time step of the
      sequence. The task amounts to having to remember a categorical sequence of
      length 10, for T time steps.

      A simple baseline can be established by considering an optimal strategy when
      no memory is available, which we deem the memoryless strategy. The
      memoryless strategy would be to predict a_8 for T + 10 entries and then
      predict each of the final 10 categories from the set {a_i}_{i=0}^7
      independently and uniformly at random. The categorical cross entropy of this
      strategy is \frac{10 log(8)}{T + 20}.

      If selective is True, then shuffle the blank spaces between the array to
      copy.

    Args:
      num_examples: Number of examples to generate.
      num_categories: Number of token types. One is used as a blank token, one is
        used as a delimiter, and the remaining are used to choose from for
        copying.
      copy_len: Number of tokens to copy.
      blank_len: Number of blank tokens inbetween copy and paste.
      selective: Whether to return a selective copy task or not.
      seed: Seed for random number generator.

    Returns:
      A PyTorch dataset.
    """
    # Assign characters.
    copy_chars = torch.arange(num_categories - 2)
    blank_char = num_categories - 2
    delim_char = num_categories - 1

    # Set random seed.
    torch.manual_seed(seed)

    # Construct input sequences.
    to_copy = torch.randint(0, len(copy_chars), (num_examples, copy_len))
    blank = torch.full((num_examples, blank_len - 1), blank_char)
    delim = torch.full((num_examples, 1), delim_char)
    to_fill = torch.full((num_examples, copy_len), blank_char)

    if selective:

        def insert(row):
            indices = torch.randperm(copy_len)[: blank_len - 1]
            row[indices] = blank_char
            return row

        inputs = torch.stack([insert(row) for row in to_copy])
    else:
        inputs = torch.cat((to_copy, blank), dim=1)
    inputs = torch.cat((inputs, delim, to_fill), dim=1)

    # Construct output sequences.
    blank = torch.full((num_examples, blank_len + copy_len), blank_char)
    outputs = torch.cat((blank, to_copy), dim=1)

    # Construct dataset.
    return TensorDataset(inputs, outputs)


def generate_adding(
    num_examples: int = 5,
    sequence_len: int = 10,
    seed: int = 0,
) -> TensorDataset:
    """Generate an adding task.

    This adding task is taken from (Arjovsky, Shah, and Bengio, 2016). From their
    paper:

      We closely follow the adding problem defined in (Hochreiter & Schmidhuber,
      1997) to explain the task at hand. Each input consists of two sequences of
      length T. The first sequence, which we denote x, consists of numbers sampled
      uniformly at random U[0, 1]. The second sequence is an indicator sequence
      consisting of exactly two entries of 1 and remaining entries 0. The first 1
      entry is located uniformly at random in the first half of the sequence,
      whilst the second 1 entry is located uniformly at random in the second half.
      The output is the sum of the two entries of the first sequence,
      corresponding to where the 1 entries are located in the second sequence. A
      naive strategy of predicting 1 as the output regardless of the input
      sequence gives an expected mean squared error of 0.167, the variance of the
      sum of two independent uniform distributions. This is our baseline to beat.

    Args:
      num_examples: Number of examples to generate.
      sequence_len: Length of each sequence.
      seed: Seed for random number generator.

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

    # Concatenate the inputs.
    inputs = torch.cat((seq_1, seq_2), dim=1)

    # Construct dataset.
    return TensorDataset(inputs, outputs)


def generate_induction_heads(
    num_examples: int = 5,
    sequence_len: int = 30,
    vocab_size: int = 20,
    seed: int = 0,
) -> TensorDataset:
    """Generate an induction heads task.

    This induction heads task is taken from (Dao, Fu, Saab, 2023). From their
    paper:

      The Induction Head task tests how well a model can recall content after a
      special token. At the end of the sequence, the model must recall the token
      that appeared immediately after the special token earlier in the sequence.

    Args:
      num_examples: Number of examples to generate.
      sequence_len: Length of each sequence.
      vocab_size: Size of the vocabulary, including the special token.
      seed: Seed for random number generator.

    Returns:
      A PyTorch dataset.
    """
    # Set random seed.
    torch.manual_seed(seed)

    # Set the special token.
    special = vocab_size - 1

    inputs = torch.randint(0, vocab_size - 1, (num_examples, sequence_len))

    # Place special token somewhere before the last token.
    idx = torch.randint(0, sequence_len - 2, (num_examples,))
    inputs[torch.arange(num_examples), idx] = special

    # Place special token at the end of the sequence.
    inputs[torch.arange(num_examples), -1] = special

    outputs = inputs[torch.arange(num_examples), idx + 1]

    return TensorDataset(inputs, outputs)


def generate_associative_recall(
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
      A PyTorch dataset.
    """
    # Set random seed.
    torch.manual_seed(seed)

    idx = torch.randint(0, vocab_size, (num_examples, sequence_len // 2))

    def get_assoc(start: int, end: int):
        # Range of values
        x = torch.arange(start, end)

        # Make num_examples copies.
        x = x.repeat(num_examples, 1)

        # Shuffle each row independently.
        x = x[torch.randperm(x.size(0)), :]

        # Grab the corresponding indices
        return x[torch.arange(num_examples).unsqueeze(1), idx]

    keys = get_assoc(0, vocab_size)
    vals = get_assoc(vocab_size, 2 * vocab_size)

    # Interleave keys and values by row.
    inputs = torch.zeros((num_examples, sequence_len), dtype=keys.dtype)
    inputs[:, 0::2] = keys
    inputs[:, 1::2] = vals

    # Get key we want to find associated value for.
    idx = torch.randint(0, vocab_size, (num_examples,))
    keys = keys[torch.arange(num_examples), idx].unsqueeze(1)
    inputs = torch.cat((inputs, keys), dim=1)
    outputs = vals[torch.arange(num_examples), idx]

    return TensorDataset(inputs, outputs)
