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
import dm_ai_module
import torch.nn.functional as F

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
    # policies may be numpy array or already a tensor (depending on how the NPZ was created)
    policies_obj = data['policies']
    if isinstance(policies_obj, np.ndarray):
        policies = torch.from_numpy(policies_obj.astype('float32')).float()
    elif isinstance(policies_obj, torch.Tensor):
        policies = policies_obj.float()
    else:
        policies = torch.tensor(np.array(policies_obj, dtype=np.float32)).float()
    values = torch.from_numpy(data['values']).float()
    # Optional: legal action masks provided by data collector (boolean mask, shape [N, action_dim])
    legal_masks = None
    if 'legal_masks' in data:
        legal_masks = torch.from_numpy(data['legal_masks']).bool()
    
    print(f"  States:   {states.shape}")
    print(f"  Policies: {policies.shape}")
    print(f"  Values:   {values.shape}")
    print(f"  Total samples: {len(states)}")
    
    # Create dataset and dataloader
    if legal_masks is not None:
        dataset = TensorDataset(states, policies, values, legal_masks)
    else:
        dataset = TensorDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Decide action_dim: prefer canonical engine CommandEncoder size if available,
    # otherwise derive from dataset policies shape.
    print("\nInitializing DuelTransformer...")
    if hasattr(dm_ai_module, 'CommandEncoder') and dm_ai_module.CommandEncoder is not None:
        action_dim = dm_ai_module.CommandEncoder.TOTAL_COMMAND_SIZE
        print(f"  Using CommandEncoder.TOTAL_COMMAND_SIZE = {action_dim}")
        # Warn if dataset policies size differs
        if policies.shape[1] != action_dim:
            print(f"  Warning: dataset policy dim ({policies.shape[1]}) != CommandEncoder size ({action_dim})")
    else:
        action_dim = int(policies.shape[1])
        print(f"  Using action_dim derived from dataset: {action_dim}")

    model = DuelTransformer(
        vocab_size=1000,
        action_dim=action_dim,
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
    # Compute class weights from dataset to counteract PASS-heavy labels
    try:
        import numpy as _np
        counts = _np.sum(data['policies'], axis=0)
        # Avoid division by zero
        eps = 1e-6
        inv = (_np.sum(counts) / (counts + eps))
        inv = inv.astype('float32')
        # Normalize scale to keep magnitudes reasonable
        inv = inv / _np.mean(inv)
        class_weights = torch.tensor(inv, dtype=torch.float32).to(device)
        policy_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print('  Using class-weighted CrossEntropyLoss (fallback for hard labels)')
    except Exception:
        policy_loss_fn = nn.CrossEntropyLoss()
    # KL loss will be used when policy_batch appears to be a probability distribution (soft labels).
    kl_loss_fn = lambda log_probs, target_probs: F.kl_div(log_probs, target_probs, reduction='batchmean')
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
            # Unpack depending on whether legal_masks included
            if legal_masks is not None:
                state_batch, policy_batch, value_batch, legal_mask_batch = state_batch, policy_batch, value_batch, None
            try:
                # If dataset included legal masks then DataLoader yields 4-tuple
                if len(batch) == 4:
                    state_batch, policy_batch, value_batch, legal_mask_batch = batch
                else:
                    state_batch, policy_batch, value_batch = batch
                    legal_mask_batch = None
            except Exception:
                # Fallback (older dataloaders)
                legal_mask_batch = None

            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            value_batch = value_batch.to(device)
            if legal_mask_batch is not None:
                legal_mask_batch = legal_mask_batch.to(device)
            
            # Forward
            optimizer.zero_grad()
            # padding_mask: current heuristic is (state == 0). If your tokenizer/padding token differs,
            # replace this with the tokenizer's padding token or explicit padding mask provided by DataCollector.
            padding_mask = (state_batch == 0)
            policy_logits, value_pred = model(state_batch, padding_mask=padding_mask)
            
            # Loss
            # If legal action mask is available, disallow illegal actions by masking logits and
            # zeroing/renormalizing target probabilities on illegal positions so the model does not learn them.
            if legal_mask_batch is not None:
                # legal_mask_batch expected shape [B, action_dim], dtype=bool
                illegal_mask = ~legal_mask_batch
                policy_logits = policy_logits.masked_fill(illegal_mask, -1e9)
                # Zero out target prob mass on illegal actions and renormalize per-sample
                target_probs = policy_batch.clone()
                target_probs = target_probs * legal_mask_batch.float()
                sums = target_probs.sum(dim=1, keepdim=True)
                small = 1e-8
                needs_uniform = (sums < small).squeeze(1)
                # For samples where all target mass fell into illegal actions, replace with uniform over legal
                if needs_uniform.any():
                    for i in torch.nonzero(needs_uniform, as_tuple=False):
                        idx = i.item()
                        legal = legal_mask_batch[idx]
                        if legal.any():
                            target_probs[idx] = legal.float() / legal.float().sum()
                        else:
                            # extremely degenerate: no legal actions? leave target as-is
                            pass
                    sums = target_probs.sum(dim=1, keepdim=True)
                target_probs = target_probs / (sums + small)
            else:
                target_probs = policy_batch

            # Decide loss type: if target looks like a distribution (soft labels), use KLDiv; else CrossEntropy
            use_kl = False
            # soft label heuristic: float dtype and sums close to 1.0 and not one-hot
            if policy_batch.dtype.is_floating_point:
                row_sums = policy_batch.sum(dim=1)
                if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
                    # check whether it's not strictly one-hot
                    max_vals, _ = policy_batch.max(dim=1)
                    ones = (max_vals > 0.999)
                    if not ones.all():
                        use_kl = True

            if use_kl:
                # KLDiv expects log-probs and target probs
                log_probs = F.log_softmax(policy_logits, dim=1)
                loss_policy = kl_loss_fn(log_probs, target_probs)
                # Note: we keep CrossEntropy class_weights as fallback for hard targets
            else:
                # Hard targets: argmax then CrossEntropy
                policy_targets = torch.argmax(target_probs, dim=1)
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
    # Run artifact manager to prune/archive artifacts (best-effort)
    try:
        import subprocess, sys
        subprocess.run([sys.executable, 'training/artifact_manager.py'], check=False)
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("Training complete")
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
