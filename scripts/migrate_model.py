#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migrate old DuelTransformer checkpoints to the new dynamic output layer architecture.
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.ai.agent.transformer_model import DuelTransformer

def migrate_model(input_path: str, output_path: str, reserved_dim: int = 1024):
    print(f"Migrating model from {input_path} to {output_path}...")

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    checkpoint = torch.load(input_path, map_location='cpu')
    old_state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer config from old model if possible, or use defaults
    # Since we don't have config saved in all checkpoints, we assume standard defaults
    vocab_size = 1000
    d_model = 256
    nhead = 8
    num_layers = 6
    dim_feedforward = 1024
    max_len = 200

    # Try to detect action_dim from policy head weight
    old_weight_key = 'policy_head.1.weight'
    if old_weight_key in old_state_dict:
        old_action_dim = old_state_dict[old_weight_key].shape[0]
        print(f"Detected action_dim: {old_action_dim}")
    else:
        print("Error: Could not find policy head weights in checkpoint.")
        return

    # Initialize new model
    new_model = DuelTransformer(
        vocab_size=vocab_size,
        action_dim=old_action_dim,
        reserved_dim=reserved_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len
    )

    new_state_dict = new_model.state_dict()

    # Copy weights
    copied_keys = []
    ignored_keys = []

    for key, value in old_state_dict.items():
        if 'policy_head' in key:
            continue # Handle separately
        if key in new_state_dict:
            if new_state_dict[key].shape == value.shape:
                new_state_dict[key] = value
                copied_keys.append(key)
            else:
                print(f"Warning: Shape mismatch for {key}: {value.shape} vs {new_state_dict[key].shape}")
                ignored_keys.append(key)
        else:
            ignored_keys.append(key)

    # Handle Policy Head
    # Assumes nn.Sequential(LayerNorm, Linear) -> policy_head.1 is Linear
    old_weight = old_state_dict.get('policy_head.1.weight')
    old_bias = old_state_dict.get('policy_head.1.bias')

    if old_weight is not None:
        new_weight = new_state_dict['policy_head.1.weight']
        # Copy slice
        new_weight[:old_action_dim] = old_weight
        new_state_dict['policy_head.1.weight'] = new_weight
        print(f"Copied policy weights: {old_weight.shape} -> {new_weight.shape}")

    if old_bias is not None:
        new_bias = new_state_dict['policy_head.1.bias']
        new_bias[:old_action_dim] = old_bias
        new_state_dict['policy_head.1.bias'] = new_bias
        print(f"Copied policy bias: {old_bias.shape} -> {new_bias.shape}")

    # Load updated state dict
    new_model.load_state_dict(new_state_dict)

    # Save
    save_dict = {
        'model_state_dict': new_model.state_dict(),
        'epoch': checkpoint.get('epoch', 0),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict', None),
        'action_dim': old_action_dim,
        'reserved_dim': reserved_dim
    }

    torch.save(save_dict, output_path)
    print(f"Successfully saved migrated model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate DuelTransformer checkpoint to dynamic output layer")
    parser.add_argument("input", help="Path to input checkpoint")
    parser.add_argument("output", help="Path to output checkpoint")
    parser.add_argument("--reserved-dim", type=int, default=1024, help="New reserved dimension size")

    args = parser.parse_args()
    migrate_model(args.input, args.output, args.reserved_dim)
