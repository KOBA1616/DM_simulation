#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Transformer Training
シンプルな強化学習スクリプト

Executed on: 2026-01-18
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを設定してパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# カレントディレクトリをプロジェクトルートに変更（データ読み込みのため）
os.chdir(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from datetime import datetime

from dm_toolkit.ai.agent.transformer_model import DuelTransformer

def train_simple(data_path: str = "data/transformer_training_data.npz", epochs: int = 3, batch_size: int = 8):
    """Simple training loop for DuelTransformer"""
    
    print("=" * 80)
    print("SIMPLE TRANSFORMER TRAINING")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"ERROR: File not found: {data_path}")
        return
    
    data = np.load(data_path)
    states = torch.from_numpy(data['states']).long()
    policies = torch.from_numpy(data['policies']).float()
    values = torch.from_numpy(data['values']).float()
    
    print(f"  States:   {states.shape}")
    print(f"  Policies: {policies.shape}")
    print(f"  Values:   {values.shape}")
    print(f"  Total samples: {len(states)}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    print("\nInitializing DuelTransformer...")
    model = DuelTransformer(
        vocab_size=1000,
        action_dim=600,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        max_len=200
    ).to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for {epochs} epoch(s)...")
    print("-" * 80)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        
        for batch_idx, (state_batch, policy_batch, value_batch) in enumerate(dataloader):
            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            value_batch = value_batch.to(device)
            
            # Forward
            optimizer.zero_grad()
            padding_mask = (state_batch == 0)
            policy_logits, value_pred = model(state_batch, padding_mask=padding_mask)
            
            # Loss
            policy_targets = torch.argmax(policy_batch, dim=1)
            loss_policy = policy_loss_fn(policy_logits, policy_targets)
            loss_value = value_loss_fn(value_pred, value_batch)
            loss_total = loss_policy + loss_value
            
            # Backward
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Stats
            epoch_loss += loss_total.item()
            epoch_policy_loss += loss_policy.item()
            epoch_value_loss += loss_value.item()
            num_batches += 1
            
            if (batch_idx + 1) % max(1, len(dataloader) // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss_total.item():.4f} (P: {loss_policy.item():.4f}, V: {loss_value.item():.4f})")
        
        avg_loss = epoch_loss / num_batches
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Avg Value Loss: {avg_value_loss:.4f}")
    
    # Save model
    print("\nSaving model...")
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(checkpoint_dir, f"duel_transformer_{timestamp}.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
    }, checkpoint_path)
    print(f"  Saved to {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("✓ Training complete")
    print(f"\nNext steps:")
    print(f"  1. Generate more training data: python generate_training_data.py")
    print(f"  2. Run another training round")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/transformer_training_data.npz")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    train_simple(args.data_path, args.epochs, args.batch_size)
