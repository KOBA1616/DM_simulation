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
            print("  Suggestion: run training/convert_training_policies.py to convert datasets to the canonical CommandEncoder mapping.")
    else:
        action_dim = int(policies.shape[1])
        print(f"  Using action_dim derived from dataset: {action_dim}")

    # Idea 2: Reserved Dimension for Dynamic Output Layer
    reserved_dim = 1024

    model = DuelTransformer(
        vocab_size=1000,
        action_dim=action_dim,
        reserved_dim=reserved_dim,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        max_len=200
    ).to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Action Dim: {action_dim}, Reserved Dim: {reserved_dim}")
    
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
        # We need to pad class_weights if action_dim < reserved_dim
        if len(class_weights) < reserved_dim:
            pad_size = reserved_dim - len(class_weights)
            # Pad with 1.0 (neutral weight) or 0.0?
            # Since these classes shouldn't appear in data, weight doesn't matter much unless model predicts them.
            # But model masks them out.
            class_weights = torch.cat([class_weights, torch.ones(pad_size, device=device)])

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
        
        for batch_idx, batch in enumerate(dataloader):
             # Unpack depending on whether legal_masks included
            try:
                if len(batch) == 4:
                    state_batch, policy_batch, value_batch, legal_mask_batch = batch
                else:
                    state_batch, policy_batch, value_batch = batch
                    legal_mask_batch = None
            except Exception:
                # Fallback
                legal_mask_batch = None
                if len(batch) >= 3:
                     state_batch, policy_batch, value_batch = batch[:3]

            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            value_batch = value_batch.to(device)
            if legal_mask_batch is not None:
                legal_mask_batch = legal_mask_batch.to(device)
            
            # Pad inputs to reserved_dim if necessary
            if policy_batch.shape[1] < reserved_dim:
                pad_size = reserved_dim - policy_batch.shape[1]
                # Pad policies with 0.0
                policy_batch = torch.cat([policy_batch, torch.zeros(policy_batch.shape[0], pad_size, dtype=policy_batch.dtype, device=device)], dim=1)

            if legal_mask_batch is not None and legal_mask_batch.shape[1] < reserved_dim:
                 pad_size = reserved_dim - legal_mask_batch.shape[1]
                 # Pad mask with False (illegal)
                 legal_mask_batch = torch.cat([legal_mask_batch, torch.zeros(legal_mask_batch.shape[0], pad_size, dtype=torch.bool, device=device)], dim=1)

            # Forward
            optimizer.zero_grad()
            # padding_mask: current heuristic is (state == 0).
            padding_mask = (state_batch == 0)
            
            # Pass legal_action_mask to model for internal masking
            policy_logits, value_pred = model(state_batch, padding_mask=padding_mask, legal_action_mask=legal_mask_batch)

            # Loss Preparation
            target_probs = policy_batch.clone()

            if legal_mask_batch is not None:
                 # Ensure target distribution respects mask (zero out illegal actions in target)
                 target_probs = target_probs * legal_mask_batch.float()

                 # Renormalize
                 sums = target_probs.sum(dim=1, keepdim=True)
                 small = 1e-8
                 needs_uniform = (sums < small).squeeze(1)

                 # If sample has no legal actions in target (data inconsistency or all filtered), fix it
                 if needs_uniform.any():
                     for i in torch.nonzero(needs_uniform, as_tuple=False):
                         idx = i.item()
                         legal = legal_mask_batch[idx]
                         if legal.any():
                             target_probs[idx] = legal.float() / legal.float().sum()
                         else:
                             pass
                     sums = target_probs.sum(dim=1, keepdim=True)
                 target_probs = target_probs / (sums + small)

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
                # policy_logits are already masked with -inf, so log_softmax is correct
                log_probs = F.log_softmax(policy_logits, dim=1)
                loss_policy = kl_loss_fn(log_probs, target_probs)
            else:
                # Hard targets: argmax then CrossEntropy
                # Note: target_probs has been renormalized, so argmax is valid
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
        'action_dim': action_dim,
        'reserved_dim': reserved_dim
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
