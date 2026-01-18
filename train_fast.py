#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Training Script - No Synergy Bias
高速訓練スクリプト（Synergy Biasなし）
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

from dm_toolkit.ai.agent.transformer_model import DuelTransformer

def main():
    parser = argparse.ArgumentParser(description='Fast RL Training')
    parser.add_argument('--data', type=str, default='data/simple_training_data.npz')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FAST REINFORCEMENT LEARNING TRAINING")
    print("=== 高速訓練（Synergy Biasなし） ===")
    print("="*60 + "\n")
    
    # Device
    device = torch.device('cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print(f"Loading: {args.data}")
    data = np.load(args.data)
    states = torch.LongTensor(data['states']).to(device)
    policies = torch.FloatTensor(data['policies']).to(device)
    values = torch.FloatTensor(data['values']).to(device)
    
    print(f"States:   {states.shape}")
    print(f"Policies: {policies.shape}")
    print(f"Values:   {values.shape}\n")
    
    # Model (WITHOUT synergy matrix)
    print("Initializing DuelTransformer (no synergy)...")
    model = DuelTransformer(
        vocab_size=1000,
        action_dim=591,
        d_model=256,
        nhead=8,
        num_layers=6,
        synergy_matrix_path=None  # No synergy bias for speed
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Dataset
    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Training
    print("="*60)
    print(f"TRAINING FOR {args.epochs} EPOCHS")
    print("="*60 + "\n")
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_states, batch_policies, batch_values) in enumerate(loader):
            # Forward
            policy_out, value_out = model(batch_states)
            
            # Loss
            policy_loss = nn.CrossEntropyLoss()(policy_out, batch_policies)
            value_loss = nn.MSELoss()(value_out, batch_values)
            loss = policy_loss + value_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} (P:{policy_loss.item():.4f} V:{value_loss.item():.4f})")
        
        # Epoch summary
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss:        {avg_loss:.4f}")
        print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Avg Value Loss:  {avg_value_loss:.4f}\n")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path("models") / f"duel_transformer_fast_{timestamp}.pth"
    model_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_loss,
        'epochs': args.epochs,
    }, model_path)
    
    print("="*60)
    print(f"✓ Model saved: {model_path}")
    print("="*60 + "\n")
    
    # Evaluation
    print("="*60)
    print("EVALUATION")
    print("="*60 + "\n")
    
    model.eval()
    with torch.no_grad():
        # Sample prediction
        sample_states = states[:10]
        policy_pred, value_pred = model(sample_states)
        
        print("Sample predictions:")
        for i in range(min(5, len(sample_states))):
            true_val = values[i].item()
            pred_val = value_pred[i].item()
            error = abs(true_val - pred_val)
            print(f"  Sample {i+1}: True={true_val:+.2f}, Pred={pred_val:+.2f}, Error={error:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("1. Run evaluation: python evaluate_model.py")
    print("2. Generate more data: python simple_game_generator.py")
    print("3. Continue training: python train_fast.py --epochs 5")
    print()


if __name__ == '__main__':
    main()
