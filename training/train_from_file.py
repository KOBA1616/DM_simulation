#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic trainer: loads arbitrary training .npz and configures DuelTransformer accordingly.
"""
import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime

from dm_toolkit.ai.agent.transformer_model import DuelTransformer


def train_from_file(data_path: str = "data/simple_training_data.npz", epochs: int = 1, batch_size: int = 8):
    print("Loading data:", data_path)
    if not os.path.exists(data_path):
        print("File not found", data_path)
        return
    data = np.load(data_path)
    states = torch.from_numpy(data['states']).long()
    policies = torch.from_numpy(data['policies']).float()
    values = torch.from_numpy(data['values']).float()

    print('States shape', states.shape)
    print('Policies shape', policies.shape)
    print('Values shape', values.shape)

    vocab_size = int(states.max().item() + 1) if states.numel() > 0 else 1024
    action_dim = policies.shape[1]
    max_len = states.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    dataset = TensorDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DuelTransformer(
        vocab_size=max(1024, vocab_size),
        action_dim=action_dim,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        max_len=max_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for state_batch, policy_batch, value_batch in dataloader:
            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            value_batch = value_batch.to(device)

            optimizer.zero_grad()
            padding_mask = (state_batch == 0)
            policy_logits, value_pred = model(state_batch, padding_mask=padding_mask)

            policy_targets = torch.argmax(policy_batch, dim=1)
            loss_policy = policy_loss_fn(policy_logits, policy_targets)
            loss_value = value_loss_fn(value_pred, value_batch)
            loss_total = loss_policy + loss_value

            loss_total.backward()
            optimizer.step()

    # Save small checkpoint
    ckpt_dir = 'models'
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'duel_transformer_smoke.pth')
    torch.save({'model_state_dict': model.state_dict()}, path)
    print('Saved checkpoint to', path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/simple_training_data.npz')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    train_from_file(args.data, args.epochs, args.batch_size)
