
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from typing import List, Optional

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')

if bin_path not in sys.path:
    sys.path.append(bin_path)

from dm_toolkit.ai.agent.network import AlphaZeroNetwork

class Trainer:
    def __init__(self, data_files: List[str], model_path=None, save_path="model.pth"):
        self.save_path = save_path

        # Load multiple data files
        all_states = []
        all_policies = []
        all_values = []
        all_masks = []

        print(f"Loading {len(data_files)} data files...")

        for f in data_files:
            try:
                data = np.load(f)

                if 'states_masked' in data:
                    s = data['states_masked']
                elif 'states' in data:
                    s = data['states']
                else:
                    print(f"Skipping {f}: no states found")
                    continue

                p = data['policies']
                v = data['values']

                all_states.append(s)
                all_policies.append(p)
                all_values.append(v)

                if 'masks' in data:
                    all_masks.append(data['masks'])
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not all_states:
            raise ValueError("No valid data loaded")

        self.states = torch.tensor(np.concatenate(all_states), dtype=torch.float32)
        self.policies = torch.tensor(np.concatenate(all_policies), dtype=torch.float32)
        self.values = torch.tensor(np.concatenate(all_values), dtype=torch.float32).unsqueeze(1)

        if all_masks:
            self.masks = torch.tensor(np.concatenate(all_masks), dtype=torch.float32)
            print("Action masks loaded.")
        else:
            self.masks = None
            print("No action masks found.")

        self.input_size = self.states.shape[1]
        self.action_size = self.policies.shape[1]
        print(f"Total Data: {len(self.states)} samples. Input={self.input_size}, Action={self.action_size}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.network = AlphaZeroNetwork(self.input_size, self.action_size).to(self.device)
        if model_path and os.path.exists(model_path):
            self.network.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Loaded existing model.")

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def train(self, epochs=10, batch_size=64):
        self.network.train()
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)

        # Move data to device? (Careful with VRAM)
        # For small datasets, yes. For large, do it in batch.
        # Assuming fits in CPU RAM, move batches to GPU.

        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            batches = 0

            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]

                batch_states = self.states[batch_indices].to(self.device)
                batch_target_policies = self.policies[batch_indices].to(self.device)
                batch_target_values = self.values[batch_indices].to(self.device)

                batch_masks = None
                if self.masks is not None:
                     batch_masks = self.masks[batch_indices].to(self.device)

                pred_policies, pred_values = self.network(batch_states)

                # Value Loss: MSE
                value_loss = F.mse_loss(pred_values, batch_target_values)

                # Action Masking Logic
                if batch_masks is not None:
                    fill_mask = (batch_masks == 0).bool()
                    pred_policies = pred_policies.masked_fill(fill_mask, -1e9)

                # Policy Loss: Cross Entropy
                log_softmax = F.log_softmax(pred_policies, dim=1)
                policy_loss = -torch.sum(batch_target_policies * log_softmax) / len(batch_indices)

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
