"""
FSSG-Net: Fast-Slow Simple-Gated Networks
==========================================

A dual-timescale architecture for continual learning with simple MLP.

After ablation study (Nov 9, 2025): Simple MLP outperforms similarity-gating!
    - NN1-Simple: 89.1% retention
    - NN1-Similarity: 87.9% retention

Modules:
    models: Core FSSG-Net implementation (NN1, NN2, training utilities)

Usage:
    from src.models import NN1_SimpleMLP, NN2_ConsolidationNet
    from src.models import ReplayBuffer, train_task_with_replay
"""

__version__ = "0.2.0"
__author__ = "Mohamed El Gorrim"
__all__ = ["models"]
