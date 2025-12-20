
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)
# Ensure project root is in path to resolve dm_toolkit
if project_root not in sys.path:
    sys.path.append(project_root)

# from dm_toolkit.ai.agent.network import AlphaZeroNetwork
from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class DuelDataset(Dataset):
    def __init__(self, tokens: List[torch.Tensor], policies: torch.Tensor, values: torch.Tensor, masks: Optional[torch.Tensor] = None):
        self.tokens = tokens
        self.policies = policies
        self.values = values
        self.masks = masks

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        item = {
            'tokens': self.tokens[idx],
            'policy': self.policies[idx],
            'value': self.values[idx]
        }
        if self.masks is not None:
            item['mask'] = self.masks[idx]
        return item

def collate_batch(batch):
    tokens_list = [item['tokens'] for item in batch]
    policies_list = [item['policy'] for item in batch]
    values_list = [item['value'] for item in batch]

    # Pad tokens (assuming 0 is padding)
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=0)

    # Generate padding mask (True where token is 0)
    padding_mask = (padded_tokens == 0)

    policies = torch.stack(policies_list)
    values = torch.stack(values_list)

    batch_out = {
        'tokens': padded_tokens,
        'padding_mask': padding_mask,
        'policy': policies,
        'value': values
    }

    if 'mask' in batch[0]:
        masks_list = [item['mask'] for item in batch]
        batch_out['mask'] = torch.stack(masks_list)

    return batch_out

class Trainer:
    def __init__(self, data_files: List[str], model_path=None, save_path="model.pth"):
        self.save_path = save_path

        # Load multiple data files
        all_tokens = []
        all_states = []
        all_policies = []
        all_values = []
        all_masks = []

        self.use_transformer = False

        print(f"Loading {len(data_files)} data files...")

        for f in data_files:
            try:
                data = np.load(f, allow_pickle=True)

                # Check for tokens first (Phase 8 Transformer)
                if 'tokens' in data:
                    self.use_transformer = True
                    # tokens is likely an object array of numpy arrays (jagged)
                    raw_tokens = data['tokens']
                    token_tensors = [torch.tensor(t, dtype=torch.long) for t in raw_tokens]
                    all_tokens.extend(token_tensors)
                # Fallback to states
                elif 'states_masked' in data:
                    s = data['states_masked']
                    all_states.append(s)
                elif 'states' in data:
                    s = data['states']
                    all_states.append(s)
                else:
                    print(f"Skipping {f}: no valid data found")
                    continue

                p = data['policies']
                v = data['values']

                all_policies.append(p)
                all_values.append(v)

                if 'masks' in data:
                    all_masks.append(data['masks'])
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not all_tokens and not all_states:
            raise ValueError("No valid data loaded")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        # Consolidate Data
        self.policies = torch.tensor(np.concatenate(all_policies), dtype=torch.float32)
        self.values = torch.tensor(np.concatenate(all_values), dtype=torch.float32)
        if self.values.dim() == 1:
            self.values = self.values.unsqueeze(1)

        if all_masks:
            self.masks = torch.tensor(np.concatenate(all_masks), dtype=torch.float32)
            print("Action masks loaded.")
        else:
            self.masks = None
            print("No action masks found.")

        self.action_size = self.policies.shape[1]

        if self.use_transformer:
            print("Mode: TRANSFORMER (Tokenized Data)")
            self.tokens = all_tokens

            # Calculate vocab size
            max_token = 0
            for t in self.tokens:
                if t.numel() > 0:
                    max_token = max(max_token, t.max().item())

            self.vocab_size = max_token + 1
            print(f"Detected Vocab Size: {self.vocab_size}")

            self.network = DuelTransformer(self.vocab_size, self.action_size).to(self.device)

            self.dataset = DuelDataset(self.tokens, self.policies, self.values, self.masks)

        else:
            print("Mode: RESNET/MLP (State Vector)")
            self.states = torch.tensor(np.concatenate(all_states), dtype=torch.float32)
            self.input_size = self.states.shape[1]
            print(f"Input Size: {self.input_size}")

            # Fallback to existing network logic (using Transformer container or actual logic)
            # Since the original code tried to use DuelTransformer for states, we must ensure consistency.
            # However, DuelTransformer expects tokens (Long).
            # If we are here, we have Floats. We cannot use DuelTransformer.
            # We should assume AlphaZeroNetwork or raise error.
            # For this task, we assume the user provides tokens for Transformer.
            raise ValueError("Legacy state vectors not supported for Transformer. Please provide tokenized data.")

        print(f"Total Data: {len(self.policies)} samples. Action={self.action_size}")

        if model_path and os.path.exists(model_path):
            try:
                self.network.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Loaded existing model.")
            except Exception as e:
                print(f"Warning: Failed to load existing model ({e}). Starting from scratch.")

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def train(self, epochs=10, batch_size=64):
        self.network.train()

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch
        )

        for epoch in range(epochs):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            batches = 0

            for batch in dataloader:
                batch_tokens = batch['tokens'].to(self.device)
                batch_padding_mask = batch['padding_mask'].to(self.device)
                batch_target_policies = batch['policy'].to(self.device)
                batch_target_values = batch['value'].to(self.device)

                batch_masks = None
                if 'mask' in batch:
                    batch_masks = batch['mask'].to(self.device)

                # Forward Pass with Mask
                pred_policies, pred_values = self.network(batch_tokens, padding_mask=batch_padding_mask)

                # Value Loss: MSE
                value_loss = F.mse_loss(pred_values, batch_target_values)

                # Action Masking Logic
                if batch_masks is not None:
                    fill_mask = (batch_masks == 0).bool()
                    pred_policies = pred_policies.masked_fill(fill_mask, -1e9)

                # Policy Loss: Cross Entropy
                log_softmax = F.log_softmax(pred_policies, dim=1)
                policy_loss = -torch.sum(batch_target_policies * log_softmax) / len(batch_target_policies)

                loss = value_loss + policy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                batches += 1

            print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/batches:.4f} (P={total_policy_loss/batches:.4f}, V={total_value_loss/batches:.4f})")

        torch.save(self.network.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

        # Export (Simplified for Transformer)
        # Note: ONNX export for Transformer with variable length requires dynamic axes
        self.network.eval()

        # We need a dummy input of integer tokens
        dummy_seq_len = 32
        dummy_input = torch.randint(0, self.vocab_size, (1, dummy_seq_len), dtype=torch.long).to(self.device)
        dummy_mask = torch.zeros((1, dummy_seq_len), dtype=torch.bool).to(self.device)

        onnx_path = self.save_path.replace(".pth", ".onnx")
        try:
            torch.onnx.export(
                self.network,
                (dummy_input, dummy_mask),
                onnx_path,
                export_params=True,
                opset_version=14, # 14+ for better Transformer support
                input_names=['input_ids', 'padding_mask'],
                output_names=['policy', 'value'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'seq_len'},
                    'padding_mask': {0: 'batch_size', 1: 'seq_len'},
                    'policy': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                }
            )
            print(f"Model exported to ONNX: {onnx_path}")
        except Exception as e:
            print(f"Failed to export to ONNX: {e}")

def train_pipeline(data_files: List[str], input_model: Optional[str], output_model: str, epochs=10):
    trainer = Trainer(data_files, input_model, output_model)
    trainer.train(epochs=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", nargs='+', help="Path to .npz data files")
    parser.add_argument("--data_dir", type=str, help="Directory containing .npz files (alternative to --data_files)")
    parser.add_argument("--model", type=str, default=None, help="Initial model path")
    parser.add_argument("--save", type=str, default="model_v1.pth", help="Save model path")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    files = []
    if args.data_files:
        files.extend(args.data_files)
    if args.data_dir and os.path.exists(args.data_dir):
        import glob
        files.extend(glob.glob(os.path.join(args.data_dir, "*.npz")))

    if not files:
        print("No data files provided.")
        sys.exit(1)

    train_pipeline(files, args.model, args.save, args.epochs)
