# Spectral State Space Models 🔢🔄🔢

## Overview

This repository contains a PyTorch implementation for training and evaluating Spectral State Space Models (SSMs), as described in the paper [Spectral State Space Models](https://arxiv.org/abs/2312.06837).

The paper studies sequence modeling for prediction tasks with long range
dependencies and proposes a new formulation for state space models (SSMs) based
on learning linear dynamical systems with the spectral filtering algorithm from Hazan et al. (2017). This gives rise to a novel sequence prediction
architecture: the spectral state space model.

Spectral SSMs offer two primary advantages:
1. **Provable robustness**: Their performance is independent of the spectrum of the underlying dynamics and the dimensionality of the problem.
2. **Efficient learning**: These models use fixed convolutional filters that don't require learning, yet outperform traditional SSMs in both theory and practice.

## Key Features

- PyTorch implementation of the Spectral State Space Model
- Full (distributed) training pipeline for training and evaluation
- Includes implementations of various other architectures to run benchmarks against (Transformer, Mamba, Samba, etc.)
- Support for training on CPUs, GPUs, and Apple's Metal Performance Shaders (MPS) backend

## Installation

We recommend using the [uv](https://github.com/astral-sh/uv) package installer and resolver from Charlie Marsh '15 and the rest of the Astral team.

1. Clone the repository and navigate to the `spectral_ssm` directory:

```bash
git clone https://github.com/windsornguyen/spectral_ssm.git
cd spectral_ssm
```

2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv ssm_env
source ssm_env/bin/activate
```

3. If you plan to use Apple's MPS backend, install PyTorch Nightly first:

```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

4. Install the required packages:

```bash
uv pip install -r requirements.txt
```

Or, for an editable installation:

```bash
uv pip install -e .
```

## Usage

The main training pipeline is contained in `example.py`. To run it:

```bash
torchrun --nproc_per_node=1 example.py
```

For more detailed usage instructions, please refer to the `docs/` directory. (TODO: To be made + document distributed training procedure)

## Model Architecture

The core of our implementation is the Spectral Temporal Unit (STU) block, which can be found in `model.py`. This file also contains the complete model architecture.

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to submit issues, feature requests, and pull requests.

## Citing This Work

If you use this implementation in your research, please cite the following [paper](https://arxiv.org/abs/2312.06837):
```bibtex
@misc{agarwal2024spectralstatespacemodels,
      title={Spectral State Space Models}, 
      author={Naman Agarwal and Daniel Suo and Xinyi Chen and Elad Hazan},
      year={2024},
      eprint={2312.06837},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2312.06837}, 
}
```

## Acknowledgments

- [Daniel Suo](https://github.com/danielsuo) for the original JAX [implementation](https://github.com/google-deepmind/spectral_ssm)
- The original authors of the Spectral State Space Models paper
- The open-source community for their invaluable contributions to the field of machine learning
- Windsor Nguyen, Isabel Liu, Yagiz Devre, and Evan Dogariu for their work on this project in summer 2024

## License
Copyright 2024 Windsor Nguyen, Isabel Liu, Yagiz Devre, Evan Dogariu

This project is licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. In simple terms:

You're free to use, modify, and distribute this software.
If you modify it, please indicate what changes you've made.
You must include the license and copyright notice with any distribution.
This software comes with no warranty.
We're not liable for any damages from its use.

For the full license text, see the LICENSE file in this repository.
