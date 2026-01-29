#!/usr/bin/env python3
"""Fine-tune only the policy head to canonical action dim.

Usage: python training/fine_tune_policy_head.py <checkpoint.pth> --data data/transformer_training_data_converted.npz
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dm_toolkit.ai.agent.transformer_model import DuelTransformer


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.states = torch.from_numpy(d['states']).long()
        self.policies = torch.from_numpy(d['policies']).float()
        self.values = torch.from_numpy(d['values']).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def reconcile_and_load(model: DuelTransformer, state: dict):
    # Attempt to load matching keys; if policy head shape differs, copy overlapping rows
    model_state = model.state_dict()
    new_state = dict(state)
    # Find policy weight/bias keys in checkpoint if present
    ck_w_key = None
    ck_b_key = None
    for k in new_state.keys():
        if k.endswith('policy_head.1.weight'):
            ck_w_key = k
        if k.endswith('policy_head.1.bias'):
            ck_b_key = k

    tgt_w_key = None
    tgt_b_key = None
    for k in model_state.keys():
        if k.endswith('policy_head.1.weight'):
            tgt_w_key = k
        if k.endswith('policy_head.1.bias'):
            tgt_b_key = k

    if ck_w_key and tgt_w_key and ck_w_key in new_state:
        ck_w = new_state[ck_w_key]
        tgt_w = model_state[tgt_w_key]
        # Copy overlapping rows
        min_out = min(ck_w.shape[0], tgt_w.shape[0])
        min_in = min(ck_w.shape[1], tgt_w.shape[1])
        import torch as _torch
        new_w = _torch.zeros_like(tgt_w)
        new_w[:min_out, :min_in] = ck_w[:min_out, :min_in]
        new_state[tgt_w_key] = new_w
    if ck_b_key and tgt_b_key and ck_b_key in new_state:
        ck_b = new_state[ck_b_key]
        tgt_b = model_state[tgt_b_key]
        import torch as _torch
        new_b = _torch.zeros_like(tgt_b)
        new_b[:min(len(ck_b), len(tgt_b))] = ck_b[:min(len(ck_b), len(tgt_b))]
        new_state[tgt_b_key] = new_b

    # Load remaining keys permissively
    model.load_state_dict(new_state, strict=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=str)
    p.add_argument('--data', type=str, default='data/transformer_training_data_converted.npz')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--out', type=str, default='models/checkpoints/policy_finetuned.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Determine desired action_dim from CommandEncoder if available
    try:
        import dm_ai_module as _dm
        desired = int(_dm.CommandEncoder.TOTAL_COMMAND_SIZE)
    except Exception:
        # Fallback to dataset policy length
        tmp = np.load(args.data)
        desired = int(tmp['policies'].shape[1])

    print('Desired action_dim:', desired)

    # Create model with desired action_dim
    model = DuelTransformer(vocab_size=1000, action_dim=desired, d_model=256, nhead=8, num_layers=6, max_len=200).to(device)

    # Load checkpoint
    ck = torch.load(args.checkpoint, map_location='cpu')
    state = ck.get('model_state_dict', ck)
    reconcile_and_load(model, state)

    # Freeze all params except policy head
    for name, p in model.named_parameters():
        p.requires_grad = False
    for name, p in model.policy_head.named_parameters():
        p.requires_grad = True

    dataset = SimpleDataset(args.data)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([p for p in model.policy_head.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for states, policies, values in dl:
            states = states.to(device)
            target_idx = torch.argmax(policies, dim=1).to(device)
            padding_mask = (states == 0)
            optimizer.zero_grad()
            logits, _ = model(states, padding_mask=padding_mask)
            loss = loss_fn(logits, target_idx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch} loss {total_loss/len(dl):.4f}')

    # Save checkpoint
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, args.out)
    print('Saved fine-tuned model to', args.out)


if __name__ == '__main__':
    main()
