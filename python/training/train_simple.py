import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any, Optional

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import models
try:
    from dm_toolkit.ai.agent.network import AlphaZeroTransformer, AlphaZeroMLP
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

class TransformerDataset(Dataset):
    """
    Dataset for handling variable-length token sequences.
    """
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found.")

        print(f"Loading data from {data_path}...")
        data = np.load(data_path, allow_pickle=True)

        # Check if we have tokens or states
        if 'tokens' in data:
            self.mode = 'tokens'
            # data['tokens'] is an object array of arrays
            self.sequences = data['tokens']
        elif 'states' in data:
            self.mode = 'states'
            self.states = data['states']
        else:
            raise ValueError(f"No valid data ('tokens' or 'states') found in {data_path}")

        self.policies = data['policies']
        self.values = data['values']

        print(f"Loaded {len(self.policies)} samples.")

        # Calculate vocab size if in token mode
        self.vocab_size = 0
        if self.mode == 'tokens':
            max_token = 0
            # Check a sample to estimate or iterate all if fast enough
            if len(self.sequences) > 0:
                # heuristic: check first 1000 samples
                for i in range(min(len(self.sequences), 1000)):
                    if len(self.sequences[i]) > 0:
                        max_token = max(max_token, np.max(self.sequences[i]))
            self.vocab_size = int(max_token) + 1
            print(f"Estimated vocab size: {self.vocab_size}")

    def __len__(self):
        return len(self.policies)

    def __getitem__(self, idx):
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)

        if self.mode == 'tokens':
            seq = self.sequences[idx]
            # Convert to tensor
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            return seq_tensor, policy, value
        else:
            state = torch.tensor(self.states[idx], dtype=torch.float32)
            return state, policy, value

def collate_fn(batch):
    """
    Custom collate function to pad sequences.
    """
    inputs, policies, values = zip(*batch)

    # Check if inputs are sequences (tensors of dim 1 with varying lengths)
    if inputs[0].dim() == 1:
        # Pad sequences
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

        # Create mask (True for valid tokens, False for padding)
        lengths = torch.tensor([len(x) for x in inputs])
        max_len = padded_inputs.size(1)
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        policies = torch.stack(policies)
        values = torch.stack(values)

        return padded_inputs, mask, policies, values
    else:
        # Fixed size tensors
        inputs = torch.stack(inputs)
        policies = torch.stack(policies)
        values = torch.stack(values)
        return inputs, None, policies, values

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TransformerDataset(args.data_files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    action_size = dataset.policies.shape[1]

    model = None
    if args.network_type == 'transformer':
        # Linear Attention Transformer
        vocab_size = max(dataset.vocab_size, 10000) # Ensure at least default
        print(f"Initializing AlphaZeroTransformer (Linear Attention) with vocab={vocab_size}...")
        model = AlphaZeroTransformer(action_size, vocab_size=vocab_size).to(device)

    elif args.network_type == 'duel_transformer':
        # Standard Self-Attention Transformer
        vocab_size = max(dataset.vocab_size, 10000)
        print(f"Initializing DuelTransformer (Standard Self-Attention) with vocab={vocab_size}...")
        model = DuelTransformer(vocab_size=vocab_size, action_dim=action_size).to(device)

    elif args.network_type == 'mlp':
        input_size = dataset.states.shape[1] if hasattr(dataset, 'states') else 0
        if input_size == 0:
            raise ValueError("MLP requires 'states' (tensor data) in .npz file.")
        print(f"Initializing AlphaZeroMLP with input_size={input_size}...")
        model = AlphaZeroMLP(input_size, action_size).to(device)
    else:
        raise ValueError(f"Unknown network type: {args.network_type}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        p_loss_sum = 0
        v_loss_sum = 0

        for batch_idx, (inputs, mask, target_policies, target_values) in enumerate(dataloader):
            inputs, target_policies, target_values = inputs.to(device), target_policies.to(device), target_values.to(device)
            if mask is not None:
                mask = mask.to(device)

            optimizer.zero_grad()

            if args.network_type == 'transformer':
                 # AlphaZeroTransformer expects mask (True for valid)
                 pred_logits, pred_values = model(inputs, mask)
            elif args.network_type == 'duel_transformer':
                 # DuelTransformer expects padding_mask (True for PAD)
                 padding_mask = ~mask
                 pred_logits, pred_values = model(inputs, padding_mask)
            else:
                 # MLP
                 pred_logits, pred_values = model(inputs)

            # Policy Loss: Cross Entropy
            log_probs = torch.log_softmax(pred_logits, dim=1)
            policy_loss = -torch.sum(target_policies * log_probs) / inputs.size(0)

            # Value Loss: MSE
            value_loss = mse_loss(pred_values.squeeze(-1), target_values)

            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            p_loss_sum += policy_loss.item()
            v_loss_sum += value_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} (Pol: {p_loss_sum/len(dataloader):.4f}, Val: {v_loss_sum/len(dataloader):.4f})")

    print(f"Saving model to {args.save}")
    torch.save(model.state_dict(), args.save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", type=str, required=True, help="Path to .npz data file")
    parser.add_argument("--save", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--network_type", type=str, default="transformer", choices=["transformer", "duel_transformer", "mlp"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    train(args)
