import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import time

# Import Model
from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.synergy import SynergyGraph

class DuelDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states = torch.from_numpy(data['states']).long()
        self.policies = torch.from_numpy(data['policies']).float()
        self.values = torch.from_numpy(data['values']).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def train(args):
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 8
    LR = 1e-4
    WARMUP_STEPS = 1000
    EPOCHS = 1

    # Model Config (from Requirements)
    VOCAB_SIZE = 1000
    ACTION_DIM = 600
    D_MODEL = 256
    NHEAD = 8
    LAYERS = 6
    MAX_LEN = 200

    # 3. Data Loading
    dataset = DuelDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded dataset from {args.data_path}: {len(dataset)} samples.")

    # 4. Model Initialization
    # Q1: Manual Synergy Pairs - handled inside DuelTransformer via SynergyGraph defaults or manual load
    # Q5: Learnable Positional Embedding is default in DuelTransformer __init__
    model = DuelTransformer(
        vocab_size=VOCAB_SIZE,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=LAYERS,
        max_len=MAX_LEN,
        synergy_matrix_path=None # Using default or manual pairs if implemented in SynergyGraph
    ).to(device)

    # Load manual pairs into SynergyGraph if available
    if os.path.exists("data/synergy_pairs_v1.json"):
        print("Loading manual synergy pairs...")
        model.synergy_graph = SynergyGraph.from_manual_pairs(VOCAB_SIZE, "data/synergy_pairs_v1.json", device=str(device))

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Loss Functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # 5. Training Loop
    model.train()
    start_time = time.time()

    print("Starting training loop (Phase 4 Skeleton)...")

    step = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            # Forward
            # Create padding mask (0 is PAD)
            padding_mask = (states == 0)

            policy_logits, value_pred = model(states, padding_mask=padding_mask)

            # Loss
            loss_policy = policy_loss_fn(policy_logits, target_policies)
            loss_value = value_loss_fn(value_pred, target_values)
            loss = loss_policy + loss_value

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f} (P: {loss_policy.item():.4f}, V: {loss_value.item():.4f})")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    print("Success criteria: 1 epoch (1 batch at least) passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/transformer_training_data_dummy.npz")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Data file not found: {args.data_path}")
        print("Please run generate_transformer_training_data.py first.")
        exit(1)

    train(args)
