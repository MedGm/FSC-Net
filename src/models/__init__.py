"""
FSSG-Net Models (Simple MLP Architecture)

After ablation study revealed simple MLP > similarity-gating,
this package contains the validated simple architecture.

Key Finding (Nov 9, 2025):
    NN1-Simple: 89.1% retention
    NN1-Similarity: 87.9% retention
    Verdict: Simpler is better! (+1.2pp)
"""

from .nn1_simple import NN1_SimpleMLP, NN1_SimpleBaseline
from .nn2_consolidation import NN2_ConsolidationNet, NN2_SlowNet
from .training_utils import (
    ReplayBuffer,
    evaluate_models,
    train_task_with_replay,
    consolidate_nn2
)

__all__ = [
    # NN1 models
    'NN1_SimpleMLP',
    'NN1_SimpleBaseline',  # Alias for backward compatibility
    
    # NN2 models
    'NN2_ConsolidationNet',
    'NN2_SlowNet',  # Alias for backward compatibility
    
    # Training utilities
    'ReplayBuffer',
    'evaluate_models',
    'train_task_with_replay',
    'consolidate_nn2',
]
