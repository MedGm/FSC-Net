# FSSG-Net: Fast-Slow Similarity-Gated Networks

**A Dual-Timescale Neural Architecture for Continual Learning and Catastrophic Forgetting Mitigation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

FSSG-Net is a novel dual-timescale neural architecture that mitigates catastrophic forgetting in continual learning scenarios. The architecture consists of:

- **Fast Network (NN1)**: Rapidly adapts to new tasks through similarity-gated neuron interactions
- **Slow Network (NN2)**: Consolidates knowledge periodically via knowledge distillation, providing stable long-term retention

**State-of-the-art Performance:**
- Split-MNIST: 88.3% retention (+9.6pp over iCaRL)
- Permuted MNIST: 91.6% retention
- Rotated MNIST: 89.1% retention  
- CIFAR-10: 29.0% retention (+9.4pp improvement over NN1)

**Statistical Validation:**
- p = 0.0011 (highly significant, p < 0.01)
- 100% win rate across 5 random seeds
- Consistent superiority across all benchmarks

---

## ✨ Key Features

- **Dual-Timescale Learning**: Asynchronous update schedules for plasticity-stability balance
- **Similarity-Gated Routing**: Top-K attention mechanism for neuron-to-neuron communication
- **Dedicated Consolidation**: Periodic NN2-only training phases on replay buffer
- **Knowledge Distillation**: Cross-network learning (NN1 → NN2) for stable representations
- **Memory Efficient**: 200 samples/task (same as GEM, less than iCaRL's 1000)
- **Scalable**: Validated from MNIST grayscale to CIFAR-10 color images

---

## 📁 Repository Structure

```
FSSGNET/
├── data/                          # Datasets and archives (auto-downloaded)
│   ├── MNIST/                     # MNIST dataset
│   └── cifar-10-batches-py/       # CIFAR-10 dataset
│
├── notebooks/                     # Experiment notebooks
│   ├── ablation_nn1_architecture.ipynb
│   └── simple_mlp_experiments/
│       ├── cifar10_5seeds.ipynb
│       ├── hyperparameter_sensitivity.ipynb
│       ├── lambda_zero_investigation.ipynb
│       └── split_mnist_30seeds.ipynb
│
├── paper/                         # Manuscript and references
│   ├── fscnet_architecture.pdf
│   ├── fscnet_main.tex
│   └── references.bib
│
├── results/                       # Generated experiment artifacts
│   └── simple_mlp/
│       └── figures/               # PNG visualizations of experimental runs
│
├── src/                           # Source code
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── nn1_simple.py
│       ├── nn2_consolidation.py
│       └── training_utils.py
│
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore rules

```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB RAM minimum

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/MedGm/FSSGNET.git
cd FSSGNET
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

Core packages:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- pandas >= 2.0.0
- scipy >= 1.10.0

See `requirements.txt` for complete list.

---

## 🚀 Quick Start

### Running Experiments

**1. Architecture Ablation (NN1 focus):**
```bash
jupyter notebook notebooks/ablation_nn1_architecture.ipynb
```
Investigates the transition from similarity-gated routing to the simpler MLP baseline.

**2. Simple MLP experiment suite:**
Open any notebook under `notebooks/simple_mlp_experiments/` (e.g. `split_mnist_30seeds.ipynb`, `cifar10_5seeds.ipynb`, `hyperparameter_sensitivity.ipynb`, `lambda_zero_investigation.ipynb`) to reproduce the continual-learning benchmarks.

### Using the FSSG-Net API

```python
import torch
from src.models import (
  NN1_SimpleMLP,
  NN2_ConsolidationNet,
  ReplayBuffer,
  train_task_with_replay,
  consolidate_nn2,
  evaluate_models,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

nn1 = NN1_SimpleMLP(input_dim=784, hidden_dim=128, num_classes=10).to(device)
nn2 = NN2_ConsolidationNet(input_dim=784, summary_dim=64, num_classes=10).to(device)

buffer = ReplayBuffer(buffer_size_per_task=200)
opt1 = torch.optim.Adam(nn1.parameters(), lr=1e-3)
opt2 = torch.optim.Adam(nn2.parameters(), lr=5e-4)
ce_loss = torch.nn.CrossEntropyLoss()
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

for task_id, (train_loader, test_loader) in enumerate(tasks):
  # Store representative samples from the current task
  buffer.add_task(train_loader.dataset)

  train_task_with_replay(
    nn1, nn2, train_loader, buffer.get_dataset(),
    opt1, opt2, ce_loss, kl_loss, device=device
  )

  consolidate_nn2(
    nn1, nn2, buffer.get_dataset(), opt2,
    ce_loss, kl_loss, device=device
  )

  acc1, acc2 = evaluate_models(nn1, nn2, test_loader, device=device)
  print(f"Task {task_id}: NN1={acc1:.2%}, NN2={acc2:.2%}")
```

---

## 📊 Experimental Results

### Benchmark Performance

| Benchmark | Tasks | NN1 | NN2 | Improvement | Significance |
|-----------|-------|-----|-----|-------------|--------------|
| **Split-MNIST** | 5 | 85.5% | **88.3%** | +2.8pp | p=0.0011 |
| **Permuted MNIST** | 10 | 91.3% | **91.6%** | +0.3pp | - |
| **Rotated MNIST** | 5 | 88.8% | **89.1%** | +0.4pp | - |
| **CIFAR-10** | 5 | 19.6% | **29.0%** | +9.4pp | - |

### Baseline Comparisons (Split-MNIST)

| Method | Type | Memory | Retention | vs FSSG-NN2 |
|--------|------|--------|-----------|-------------|
| Naive | Fine-tuning | None | 0.0% | **+88.3pp** |
| EWC | Regularization | None | 0.0% | **+88.3pp** |
| GEM | Gradient Proj. | 200/task | 66.5% | **+21.8pp** |
| iCaRL | Distillation | 1000 total | 78.7% | **+9.6pp** |
| FSSG-NN1 | Fast Only | 200/task | 85.5% | **+2.8pp** |
| **FSSG-NN2** | **Dual-Net** | **200/task** | **88.3%** | **BEST** |

### Key Findings

1. **Task Difficulty Scaling**: NN2's advantage increases with task complexity
   - Permuted MNIST (easy): +0.3pp
   - Rotated MNIST (medium): +0.4pp
   - Split-MNIST (hard): +2.8pp
   - CIFAR-10 (hardest): **+9.4pp** ⭐

2. **Consolidation is Critical**: Without dedicated consolidation phase, NN2 < NN1 (84.6% vs 85.5%)

3. **Replay is Essential**: Without experience replay, both NN1 and NN2 achieve 0% retention

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{elgorrim2025fssgnet,
  title={Fast-Slow Similarity-Gated Networks (FSSG-Net): A Dual-Timescale Neural Architecture for Continual Learning and Catastrophic Forgetting Mitigation},
  author={El Gorrim, Mohamed},
  year={2025},
  note={Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

This work was inspired by:
- Complementary Learning Systems Theory (McClelland et al., 1995)
- iCaRL (Rebuffi et al., 2017)
- GEM (Lopez-Paz & Ranzato, 2017)
- SlowFast Networks (Feichtenhofer et al., 2019)

---

## 📧 Contact

For questions or collaborations:
- **Author**: Mohamed El Gorrim
- **Email**: elgorrim.mohamed@etu.uae.ac.ma
- **Issues**: Please use the GitHub issue tracker

---

**Last Updated**: November 9, 2025
