"""
NN2 Consolidation Network

The slow consolidation network that learns from NN1's fast predictions.
This component was validated in the original experiments and remains unchanged.

Key insight: The consolidation methodology works regardless of NN1 architecture!
"""

import torch
import torch.nn as nn


class NN2_ConsolidationNet(nn.Module):
    """
    Slow consolidation network for long-term memory
    
    This network learns from NN1's fast predictions through knowledge
    distillation and provides stable long-term retention.
    
    Architecture:
        - Takes raw input + NN1 summary as input
        - Deeper network with dropout for regularization
        - Trained with distillation loss from NN1
    
    Args:
        in_dim (int): Input dimension (default: 784 for 28x28 MNIST)
        summary_dim (int): NN1 summary dimension (default: 64)
        num_classes (int): Number of output classes (default: 10)
    
    Returns:
        logits: Class predictions [B, num_classes]
    
    Example:
        >>> nn1 = NN1_SimpleMLP(in_dim=784, neuron_dim=64, num_classes=10)
        >>> nn2 = NN2_ConsolidationNet(in_dim=784, summary_dim=64,
        ...                             num_classes=10)
        >>> logits1, summary = nn1(images)
        >>> logits2 = nn2(images, summary)
    """
    
    def __init__(self, in_dim=784, summary_dim=64, num_classes=10):
        super().__init__()
        
        # Input: raw pixels + NN1 summary
        input_size = in_dim + summary_dim
        
        self.net = nn.Sequential(
            # Layer 1: Joint processing
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            
            # Layer 2: Hidden processing
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            
            # Layer 3: Classification
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, summary):
        """
        Forward pass through consolidation network
        
        Args:
            x: Input images [B, C, H, W] or [B, D]
            summary: NN1 feature embedding [B, summary_dim]
        
        Returns:
            logits: Class predictions [B, num_classes]
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        
        # Concatenate raw input with NN1 summary
        inp = torch.cat([x_flat, summary], dim=-1)
        
        # Forward pass
        logits = torch.clamp(self.net(inp), -20, 20)
        
        return logits


# Alias for backward compatibility
NN2_SlowNet = NN2_ConsolidationNet


if __name__ == "__main__":
    # Test the model
    print("Testing NN2_ConsolidationNet...")
    
    from nn1_simple import NN1_SimpleMLP
    
    nn1 = NN1_SimpleMLP(in_dim=784, neuron_dim=64, num_classes=10)
    nn2 = NN2_ConsolidationNet(in_dim=784, summary_dim=64, num_classes=10)
    
    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 28, 28)
    
    logits1, summary = nn1(dummy_input)
    logits2 = nn2(dummy_input, summary)
    
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ NN1 logits shape: {logits1.shape}")
    print(f"✅ NN1 summary shape: {summary.shape}")
    print(f"✅ NN2 logits shape: {logits2.shape}")
    print(f"\nNN1 parameters: {sum(p.numel() for p in nn1.parameters()):,}")
    print(f"NN2 parameters: {sum(p.numel() for p in nn2.parameters()):,}")
    print("\n🎉 NN2_ConsolidationNet working correctly!")
