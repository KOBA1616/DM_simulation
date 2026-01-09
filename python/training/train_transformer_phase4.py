import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import time
from datetime import datetime

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

def save_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pth")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)
    print(f"Checkpoint saved: {filename}")

def calculate_policy_entropy(policy_logits):
    # Policy logits are raw outputs from Linear layer
    # Apply Softmax to get probabilities
    probs = torch.softmax(policy_logits, dim=-1)
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    # Entropy = - sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
    return entropy

def train(args):
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Hyperparameters & Config
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    LOG_DIR = args.log_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    CHECKPOINT_FREQ = args.checkpoint_freq

    # Model Config (from Requirements)
    VOCAB_SIZE = 1000
    ACTION_DIM = 600
    D_MODEL = 256
    NHEAD = 8
    LAYERS = 6
    MAX_LEN = 200

    # TensorBoard Writer
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir_run = os.path.join(LOG_DIR, current_time)
    writer = SummaryWriter(log_dir=log_dir_run)
    print(f"TensorBoard logging to: {log_dir_run}")

    # 3. Data Loading
    full_dataset = DuelDataset(args.data_path)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2 if device.type == 'cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded dataset from {args.data_path}: {total_size} total samples.")
    print(f"Train size: {train_size}, Validation size: {val_size}")

    # 4. Model Initialization
    model = DuelTransformer(
        vocab_size=VOCAB_SIZE,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=LAYERS,
        max_len=MAX_LEN,
        synergy_matrix_path=None
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
    global_step = 0
    start_time = time.time()

    print(f"Starting training loop for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, (states, target_policies, target_values) in enumerate(train_loader):
            step_start_time = time.time()

            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            # Forward
            # Create padding mask (0 is PAD)
            padding_mask = (states == 0)

            policy_logits, value_pred = model(states, padding_mask=padding_mask)

            # Loss
            # target_policies is (Batch, ActionDim) probabilities/one-hot.
            # CrossEntropyLoss expects class indices (Batch) for hard labels.
            # Since our data is one-hot (from Heuristic), we can use argmax.
            target_indices = torch.argmax(target_policies, dim=1)
            loss_policy = policy_loss_fn(policy_logits, target_indices)

            loss_value = value_loss_fn(value_pred, target_values)
            loss = loss_policy + loss_value

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (Optional but recommended for Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Metrics
            entropy = calculate_policy_entropy(policy_logits)
            batch_time = time.time() - step_start_time
            throughput = states.size(0) / batch_time

            global_step += 1

            # Logging
            if global_step % 10 == 0:
                writer.add_scalar('Loss/Total', loss.item(), global_step)
                writer.add_scalar('Loss/Policy', loss_policy.item(), global_step)
                writer.add_scalar('Loss/Value', loss_value.item(), global_step)
                writer.add_scalar('Policy/Entropy', entropy.item(), global_step)
                writer.add_scalar('System/Throughput', throughput, global_step)

                if torch.cuda.is_available():
                    vram_usage = torch.cuda.memory_allocated(device) / (1024 ** 3) # GB
                    writer.add_scalar('System/VRAM_GB', vram_usage, global_step)

                print(f"Epoch {epoch} | Step {global_step} | Loss {loss.item():.4f} (P: {loss_policy.item():.4f}, V: {loss_value.item():.4f}) | Ent: {entropy.item():.2f} | {throughput:.1f} samples/s")

            # Checkpointing
            if global_step % CHECKPOINT_FREQ == 0:
                save_checkpoint(model, optimizer, epoch, global_step, loss.item(), CHECKPOINT_DIR)

        # Validation Loop (End of Epoch)
        model.eval()
        val_loss = 0
        val_entropy = 0
        with torch.no_grad():
            for states, target_policies, target_values in val_loader:
                states = states.to(device)
                target_policies = target_policies.to(device)
                target_values = target_values.to(device)
                padding_mask = (states == 0)

                policy_logits, value_pred = model(states, padding_mask=padding_mask)
                target_indices = torch.argmax(target_policies, dim=1)
                loss_policy = policy_loss_fn(policy_logits, target_indices)
                loss_value = value_loss_fn(value_pred, target_values)
                val_loss += (loss_policy + loss_value).item()
                val_entropy += calculate_policy_entropy(policy_logits).item()

        val_loss /= len(val_loader)
        val_entropy /= len(val_loader)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Entropy', val_entropy, epoch)
        print(f"--- Epoch {epoch} Finished --- Validation Loss: {val_loss:.4f} | Validation Entropy: {val_entropy:.2f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Save Final Model
    save_checkpoint(model, optimizer, EPOCHS, global_step, val_loss, CHECKPOINT_DIR)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/transformer_training_data.npz", help="Path to training data .npz file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (expand 8->16->32->64)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default="logs/transformer", help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/transformer", help="Directory for checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="Steps between checkpoints")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Data file not found: {args.data_path}")
        print("Please run generate_transformer_training_data.py first.")
        exit(1)

    train(args)
