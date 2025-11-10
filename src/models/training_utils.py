"""
Training Utilities for FSSG-Net

Replay buffer, training loops, and consolidation functions for continual
learning experiments.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ReplayBuffer:
    """
    Experience replay buffer for continual learning
    
    Stores a subset of samples from each task to prevent catastrophic
    forgetting. Used during training to mix current task data with
    past experiences.
    
    Args:
        buffer_size_per_task (int): Number of samples to store per task
    
    Example:
        >>> buffer = ReplayBuffer(buffer_size_per_task=200)
        >>> buffer.add_task(task1_dataset)
        >>> buffer.add_task(task2_dataset)
        >>> replay_data = buffer.get_dataset()
    """
    
    def __init__(self, buffer_size_per_task=200):
        self.buffer_size = buffer_size_per_task
        self.stored_data = []
    
    def add_task(self, task_dataset):
        """Add samples from a new task to the buffer"""
        indices = random.sample(
            range(len(task_dataset)),
            min(self.buffer_size, len(task_dataset))
        )
        samples = [task_dataset[i] for i in indices]
        self.stored_data.extend(samples)
        print(f"   💾 Replay buffer: {len(self.stored_data)} total samples")
    
    def get_dataset(self):
        """Get all stored samples"""
        return self.stored_data


def evaluate_models(nn1, nn2, loader, device='cuda'):
    """
    Evaluate both NN1 and NN2 on a test set
    
    Args:
        nn1: Fast learning network
        nn2: Consolidation network
        loader: DataLoader for test set
        device: Device to use for computation
    
    Returns:
        acc1: NN1 accuracy
        acc2: NN2 accuracy
    """
    nn1.eval()
    nn2.eval()
    total = correct1 = correct2 = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Get predictions from both networks
            logits1, summary = nn1(xb)
            logits2 = nn2(xb, summary)
            
            pred1 = logits1.argmax(dim=-1)
            pred2 = logits2.argmax(dim=-1)
            
            total += yb.size(0)
            correct1 += (pred1 == yb).sum().item()
            correct2 += (pred2 == yb).sum().item()
    
    return correct1 / total, correct2 / total


def train_task_with_replay(nn1, nn2, train_loader, replay_loader,
                            opt1, opt2, ce_loss, kl_loss,
                            device='cuda',
                            epochs=5,
                            consolidation_interval=10,
                            lambda_distill=0.3,
                            temperature=2.0,
                            grad_clip=1.0,
                            replay_ratio=0.3):
    """
    Train NN1 and NN2 on a new task with replay from previous tasks
    
    Args:
        nn1: Fast learning network
        nn2: Consolidation network
        train_loader: DataLoader for current task
        replay_loader: List of samples from previous tasks
        opt1: Optimizer for NN1
        opt2: Optimizer for NN2
        ce_loss: Cross-entropy loss function
        kl_loss: KL divergence loss function
        device: Device for computation
        epochs: Number of epochs to train
        consolidation_interval: How often to update NN2 (every N batches)
        lambda_distill: Weight for distillation loss (0-1)
        temperature: Temperature for knowledge distillation
        grad_clip: Gradient clipping threshold
        replay_ratio: Probability of mixing replay data (0-1)
    """
    for epoch in range(epochs):
        nn1.train()
        nn2.train()
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            # Mix current task with replay data
            if (replay_loader is not None and len(replay_loader) > 0
                    and random.random() < replay_ratio):
                replay_batch = random.sample(
                    replay_loader,
                    min(len(xb), len(replay_loader))
                )
                replay_x = torch.stack([x for x, y in replay_batch])
                replay_x = replay_x.to(device)
                replay_y = torch.tensor([y for x, y in replay_batch])
                replay_y = replay_y.to(device)
                
                xb = torch.cat([xb, replay_x], dim=0)
                yb = torch.cat([yb, replay_y], dim=0)
            
            # NN1 update (every batch)
            logits1, summary = nn1(xb)
            loss1 = ce_loss(logits1, yb)
            
            opt1.zero_grad()
            loss1.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(nn1.parameters(), grad_clip)
            opt1.step()
            
            # NN2 update (periodic)
            if batch_idx % consolidation_interval == 0:
                logits2 = nn2(xb, summary.detach())
                loss_ce2 = ce_loss(logits2, yb)
                
                # Knowledge distillation from NN1 to NN2
                soft_teacher = F.softmax(
                    logits1.detach() / temperature,
                    dim=-1
                )
                soft_student = F.log_softmax(
                    logits2 / temperature,
                    dim=-1
                )
                loss_kl = kl_loss(soft_student, soft_teacher)
                loss_kl = loss_kl * (temperature ** 2)
                
                loss2 = (1.0 - lambda_distill) * loss_ce2
                loss2 = loss2 + lambda_distill * loss_kl
                
                if not (torch.isnan(loss2) or torch.isinf(loss2)):
                    opt2.zero_grad()
                    loss2.backward()
                    if grad_clip > 0:
                        nn2_params = nn2.parameters()
                        torch.nn.utils.clip_grad_norm_(nn2_params, grad_clip)
                    opt2.step()


def consolidate_nn2(nn1, nn2, replay_data, opt2, ce_loss, kl_loss,
                    device='cuda',
                    consolidation_epochs=2,
                    batch_size=64,
                    lambda_distill=0.5,
                    temperature=2.0,
                    grad_clip=1.0):
    """
    Consolidate NN2 knowledge by training on replay buffer
    
    This is called after each task to strengthen NN2's retention of all
    previous tasks. NN1 acts as the teacher.
    
    Args:
        nn1: Fast learning network (frozen, acts as teacher)
        nn2: Consolidation network (trainable)
        replay_data: List of samples from all previous tasks
        opt2: Optimizer for NN2
        ce_loss: Cross-entropy loss function
        kl_loss: KL divergence loss function
        device: Device for computation
        consolidation_epochs: Number of epochs for consolidation
        batch_size: Batch size for consolidation
        lambda_distill: Weight for distillation loss (0-1)
        temperature: Temperature for knowledge distillation
        grad_clip: Gradient clipping threshold
    """
    if len(replay_data) == 0:
        return
    
    print(f"   🧠 NN2 Consolidation: {len(replay_data)} samples, "
          f"{consolidation_epochs} epochs")
    
    replay_loader = DataLoader(
        replay_data,
        batch_size=batch_size,
        shuffle=True
    )
    
    nn1.eval()
    nn2.train()
    
    for epoch in range(consolidation_epochs):
        for xb, yb in replay_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Get teacher predictions from NN1 (frozen)
            with torch.no_grad():
                logits1, summary = nn1(xb)
            
            # Train NN2 to match NN1's predictions
            logits2 = nn2(xb, summary)
            loss_ce = ce_loss(logits2, yb)
            
            # Knowledge distillation
            soft_teacher = F.softmax(logits1 / temperature, dim=-1)
            soft_student = F.log_softmax(logits2 / temperature, dim=-1)
            loss_kl = kl_loss(soft_student, soft_teacher)
            loss_kl = loss_kl * (temperature ** 2)
            
            loss = (1.0 - lambda_distill) * loss_ce
            loss = loss + lambda_distill * loss_kl
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                opt2.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(nn2.parameters(),
                                                   grad_clip)
                opt2.step()


if __name__ == "__main__":
    # Test the utilities
    print("Testing training utilities...")
    
    from nn1_simple import NN1_SimpleMLP
    from nn2_consolidation import NN2_ConsolidationNet
    
    # Create dummy dataset
    dummy_data = [(torch.randn(28, 28), i % 10) for i in range(100)]
    
    # Test replay buffer
    buffer = ReplayBuffer(buffer_size_per_task=50)
    buffer.add_task(dummy_data)
    print(f"✅ ReplayBuffer: {len(buffer.get_dataset())} samples stored")
    
    # Test evaluation
    nn1 = NN1_SimpleMLP().to('cpu')
    nn2 = NN2_ConsolidationNet().to('cpu')
    loader = DataLoader(dummy_data, batch_size=32)
    
    acc1, acc2 = evaluate_models(nn1, nn2, loader, device='cpu')
    print(f"✅ Evaluation: NN1={acc1:.2%}, NN2={acc2:.2%}")
    
    print("\n🎉 Training utilities working correctly!")
