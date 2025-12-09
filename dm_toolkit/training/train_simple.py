
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)

from dm_toolkit.ai.agent.network import AlphaZeroNetwork

class Trainer:
    def __init__(self, data_file, model_path=None, save_path="model.pth"):
        self.data_file = data_file
        self.save_path = save_path

        print(f"Loading data from {data_file}...")
        data = np.load(data_file)

        # Determine if we have masked/full states (new format) or just 'states' (old)
        if 'states_masked' in data:
            self.states = torch.tensor(data['states_masked'], dtype=torch.float32)
        elif 'states' in data:
            self.states = torch.tensor(data['states'], dtype=torch.float32)
        else:
             raise ValueError("Data file must contain 'states' or 'states_masked'")

        self.policies = torch.tensor(data['policies'], dtype=torch.float32)
        self.values = torch.tensor(data['values'], dtype=torch.float32).unsqueeze(1)

        # Load Masks if available (Step C: Action Masking)
        if 'masks' in data:
            self.masks = torch.tensor(data['masks'], dtype=torch.float32)
            print("Action masks loaded.")
        else:
            self.masks = None
            print("No action masks found in data.")

        self.input_size = self.states.shape[1]
        self.action_size = self.policies.shape[1]
        print(f"Data loaded: {len(self.states)} samples. Input={self.input_size}, Action={self.action_size}")

        self.network = AlphaZeroNetwork(self.input_size, self.action_size)
        if model_path and os.path.exists(model_path):
            self.network.load_state_dict(torch.load(model_path))
            print("Loaded existing model.")

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def train(self, epochs=10, batch_size=64):
        self.network.train()
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            batches = 0

            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]

                batch_states = self.states[batch_indices]
                batch_target_policies = self.policies[batch_indices]
                batch_target_values = self.values[batch_indices]

                batch_masks = None
                if self.masks is not None:
                     batch_masks = self.masks[batch_indices]

                pred_policies, pred_values = self.network(batch_states)

                # Value Loss: MSE
                value_loss = F.mse_loss(pred_values, batch_target_values)

                # Action Masking Logic (Step C)
                # If we have masks, set the logits of invalid actions to a very small number (-inf)
                if batch_masks is not None:
                    # Make sure mask is 1 for legal, 0 for illegal
                    # (1 - mask) * -1e9
                    # Or just pred_policies[mask == 0] = -1e9 (requires cloning if leaf?)
                    # pred_policies is a tensor.
                    # We can use masked_fill

                    # Convert 0 in mask to True (to fill)
                    fill_mask = (batch_masks == 0).bool()
                    pred_policies = pred_policies.masked_fill(fill_mask, -1e9)

                # Policy Loss: Cross Entropy
                # pred_policies are logits. target is probability distribution.
                # Use KL Divergence or Cross Entropy.
                # CrossEntropyLoss expects class indices, but we have soft targets.
                # Use: - sum(target * log_softmax(pred))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, help="Path to .npz data file")
    parser.add_argument("--model", type=str, default=None, help="Initial model path")
    parser.add_argument("--save", type=str, default="model_v1.pth", help="Save model path")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    trainer = Trainer(args.data_file, args.model, args.save)
    trainer.train(epochs=args.epochs)
