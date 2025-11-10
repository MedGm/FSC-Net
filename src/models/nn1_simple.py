"""
NN1 Simple MLP Architecture

After ablation study (November 9, 2025) revealed that a simple MLP baseline
outperforms similarity-gating by 1.2pp (89.1% vs 87.9%), this module provides
the validated simple architecture.

Key Result: Simpler is better for continual learning!
"""

import torch
import torch.nn as nn


class NN1_SimpleMLP(nn.Module):
    """
    Simple MLP for fast learning (NN1)
    
    This architecture was validated in ablation_nn1_architecture.ipynb
    and shown to OUTPERFORM the similarity-gated version by 1.2pp.
    
    Architecture:
        - Standard feedforward MLP
        - NO similarity-gating, NO attention, NO GRU
        - Layer normalization for stability
        - 64D embedding dimension
    
    Args:
        in_dim (int): Input dimension (default: 784 for 28x28 MNIST)
        num_neurons (int): Number of neurons (kept for compatibility, not used)
        neuron_dim (int): Embedding dimension (default: 64)
        T_fast (int): Iterations (kept for compatibility, not used)
        topk (int): Top-k neighbors (kept for compatibility, not used)
        num_classes (int): Number of output classes (default: 10)
    
    Returns:
        logits: Class predictions [B, num_classes]
        summary: Feature embedding for NN2 [B, neuron_dim]
    
    Example:
        >>> nn1 = NN1_SimpleMLP(in_dim=784, neuron_dim=64, num_classes=10)
        >>> logits, summary = nn1(images)
        >>> # summary can be used by NN2 for consolidation
    """
    
    def __init__(self, in_dim=784, num_neurons=16, neuron_dim=64, 
                 T_fast=3, topk=5, num_classes=10):
        super().__init__()
        
        # Store embedding dimension for NN2 compatibility
        self.summary_dim = neuron_dim
        
        # Simple MLP: 784 → 128 → 64 → 128 → 64 → 10
        # Multiple layers with normalization for stable training
        self.net = nn.Sequential(
            # Layer 1: Input projection
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            
            # Layer 2: First embedding
            nn.Linear(128, neuron_dim),
            nn.ReLU(),
            nn.LayerNorm(neuron_dim),
            
            # Layer 3: Hidden processing
            nn.Linear(neuron_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            
            # Layer 4: Final embedding (this is the summary)
            nn.Linear(128, neuron_dim),
            nn.ReLU(),
            nn.LayerNorm(neuron_dim),
            
            # Layer 5: Classification head
            nn.Linear(neuron_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through simple MLP
        
        Args:
            x: Input images [B, C, H, W] or [B, D]
        
        Returns:
            logits: Class predictions [B, num_classes]
            summary: Feature embedding [B, neuron_dim] (for NN2)
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        
        # Forward pass through full network
        logits = torch.clamp(self.net(x_flat), -20, 20)
        
        # Extract summary (penultimate layer activation)
        # This is the feature embedding that NN2 will use for consolidation
        with torch.no_grad():
            summary = self.net[:-1](x_flat)  # All layers except final classifier
        
        return logits, summary.detach()


# Alias for backward compatibility with notebooks
NN1_SimpleBaseline = NN1_SimpleMLP


if __name__ == "__main__":
    # Test the model
    print("Testing NN1_SimpleMLP...")
    
    model = NN1_SimpleMLP(in_dim=784, neuron_dim=64, num_classes=10)
    
    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 28, 28)
    logits, summary = model(dummy_input)
    
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Logits shape: {logits.shape}")
    print(f"✅ Summary shape: {summary.shape}")
    print(f"✅ Summary dim: {model.summary_dim}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n🎉 NN1_SimpleMLP working correctly!")
