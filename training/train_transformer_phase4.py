import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
import numpy as np
import math
import json
import argparse
import os
import time
from datetime import datetime
import subprocess
import threading
from pathlib import Path
from typing import Optional

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
    return filename


def _find_baseline_checkpoint(checkpoint_dir: str, exclude_step: int):
    p = Path(checkpoint_dir)
    if not p.exists():
        return None
    candidates = list(p.glob('checkpoint_epoch_*_step_*.pth'))
    best = None
    best_step = -1
    for c in candidates:
        try:
            name = c.stem
            # parse `_step_` value
            parts = name.split('_')
            if 'step' in parts:
                idx = parts.index('step')
                step_val = int(parts[idx+1])
            else:
                continue
            if step_val < exclude_step and step_val > best_step:
                best = str(c)
                best_step = step_val
        except Exception:
            continue
    return best


def run_head2head_eval(checkpoint_path: str, checkpoint_dir: str, eval_games: int, eval_parallel: int, eval_use_pytorch: bool, eval_baseline: str | None):
    repo_root = Path(__file__).resolve().parents[2]
    h2h_script = repo_root / 'training' / 'head2head.py'
    if not h2h_script.exists():
        print(f"TRAIN_JSON: {json.dumps({'event':'h2h_failed','reason':'head2head script missing','path':str(h2h_script)})}")
        return

    baseline = None
    if eval_baseline:
        if os.path.exists(eval_baseline):
            baseline = eval_baseline
        else:
            print(f"TRAIN_JSON: {json.dumps({'event':'h2h_note','reason':'baseline_missing','path':eval_baseline})}")
    if baseline is None:
        baseline = _find_baseline_checkpoint(checkpoint_dir, exclude_step=0) if checkpoint_path is None else _find_baseline_checkpoint(checkpoint_dir, exclude_step=int(Path(checkpoint_path).stem.split('_')[-1]))
    if not baseline:
        print(f"TRAIN_JSON: {json.dumps({'event':'h2h_skipped','reason':'no_baseline_found'})}")
        return

    cmd = [str(Path(os.sys.executable)), str(h2h_script), str(checkpoint_path), str(baseline), '--games', str(eval_games), '--parallel', str(eval_parallel)]
    if eval_use_pytorch:
        cmd.append('--use_pytorch')

    try:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        print(f"TRAIN_JSON: {json.dumps({'event':'h2h_failed','reason':'proc_start_failed', 'error': str(e)})}")
        return

    def forward_stdout():
        try:
            if proc.stdout:
                for ln in proc.stdout:
                    ln = ln.rstrip('\n')
                    # forward raw H2H_JSON lines so GUI can parse
                    print(ln)
        except Exception as e:
            print(f"TRAIN_JSON: {json.dumps({'event':'h2h_failed','reason':'read_failed','error':str(e)})}")

    t = threading.Thread(target=forward_stdout, daemon=True)
    t.start()

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
    ACCUM_STEPS = max(1, getattr(args, 'accum_steps', 1))
    USE_AMP = bool(getattr(args, 'use_amp', False))
    WARMUP_STEPS = int(getattr(args, 'warmup_steps', 0))
    SCHEDULER = getattr(args, 'lr_scheduler', 'cosine')

    # Model Config (from Requirements)
    VOCAB_SIZE = 1000
    ACTION_DIM = None  # Determined from CommandEncoder or dataset below
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

    # Determine ACTION_DIM: prefer canonical CommandEncoder size if available
    try:
        import dm_ai_module as _dm
        if hasattr(_dm, 'CommandEncoder') and getattr(_dm.CommandEncoder, 'TOTAL_COMMAND_SIZE', None) is not None:
            desired = int(_dm.CommandEncoder.TOTAL_COMMAND_SIZE)
            if full_dataset.policies.shape[1] != desired:
                print(f"ERROR: dataset policy dim ({full_dataset.policies.shape[1]}) != CommandEncoder.TOTAL_COMMAND_SIZE ({desired}).")
                print("Please convert or regenerate your dataset so policies have length equal to the canonical CommandEncoder size.")
                raise SystemExit(1)
            ACTION_DIM = desired
            print(f"Using CommandEncoder.TOTAL_COMMAND_SIZE = {ACTION_DIM}")
        else:
            ACTION_DIM = int(full_dataset.policies.shape[1])
            print(f"Using action_dim derived from dataset: {ACTION_DIM}")
    except Exception:
        # Fall back to dataset-derived action dim
        ACTION_DIM = int(full_dataset.policies.shape[1])
        print(f"Warning: failed to read CommandEncoder.TOTAL_COMMAND_SIZE, using dataset-derived action_dim: {ACTION_DIM}")

    # Estimated steps per epoch used for ETA calculations
    effective_batch = BATCH_SIZE * ACCUM_STEPS
    steps_per_epoch = math.ceil(train_size / effective_batch) if effective_batch > 0 else 0
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

    # Optional AMP scaler
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if USE_AMP and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    # Loss Functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # LR Scheduler: warmup + cosine
    total_steps = EPOCHS * math.ceil(train_size / BATCH_SIZE)
    try:
        if SCHEDULER == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - WARMUP_STEPS))
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    except Exception:
        scheduler = None

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

            if USE_AMP and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    policy_logits, value_pred = model(states, padding_mask=padding_mask)
                    target_indices = torch.argmax(target_policies, dim=1)
                    loss_policy = policy_loss_fn(policy_logits, target_indices)
                    loss_value = value_loss_fn(value_pred, target_values)
                    loss = loss_policy + loss_value
                # Backward with scaler
                if scaler is None:
                    scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
            else:
                policy_logits, value_pred = model(states, padding_mask=padding_mask)
                target_indices = torch.argmax(target_policies, dim=1)
                loss_policy = policy_loss_fn(policy_logits, target_indices)
                loss_value = value_loss_fn(value_pred, target_values)
                loss = loss_policy + loss_value
                loss.backward()

            # Metrics
            entropy = calculate_policy_entropy(policy_logits)
            batch_time = time.time() - step_start_time
            throughput = states.size(0) / batch_time

            # Gradient accumulation and optimizer step
            do_step = ((batch_idx + 1) % ACCUM_STEPS) == 0
            # If final batch in epoch but not aligned, still step
            if (batch_idx + 1) == len(train_loader):
                do_step = True

            if do_step:
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Logging
            if global_step > 0 and global_step % 10 == 0:
                writer.add_scalar('Loss/Total', loss.item(), global_step)
                writer.add_scalar('Loss/Policy', loss_policy.item(), global_step)
                writer.add_scalar('Loss/Value', loss_value.item(), global_step)
                writer.add_scalar('Policy/Entropy', entropy.item(), global_step)
                writer.add_scalar('System/Throughput', throughput, global_step)

                if torch.cuda.is_available():
                    vram_usage = torch.cuda.memory_allocated(device) / (1024 ** 3) # GB
                    writer.add_scalar('System/VRAM_GB', vram_usage, global_step)

                # Compute ETA
                elapsed = time.time() - start_time
                avg_step = elapsed / global_step if global_step > 0 else 0.0
                remaining_steps = ((EPOCHS - epoch - 1) * steps_per_epoch) + max(0, steps_per_epoch - batch_idx - 1)
                eta_seconds = remaining_steps * avg_step
                elapsed_minutes = elapsed / 60.0
                eta_minutes = eta_seconds / 60.0

                print(f"Epoch {epoch} | Step {global_step} | Loss {loss.item():.4f} (P: {loss_policy.item():.4f}, V: {loss_value.item():.4f}) | Ent: {entropy.item():.2f} | {throughput:.1f} samples/s")
                # One-line JSON heartbeat for GUI consumption
                hb = {
                    'event': 'heartbeat',
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'global_step': global_step,
                    'elapsed_minutes': round(elapsed_minutes, 2),
                    'eta_minutes': round(eta_minutes, 2),
                    'remaining_steps': int(remaining_steps),
                    'steps_per_epoch': int(steps_per_epoch)
                }
                print('TRAIN_JSON: ' + json.dumps(hb))

            # Checkpointing
            if global_step > 0 and global_step % CHECKPOINT_FREQ == 0:
                        ckpath = save_checkpoint(model, optimizer, epoch, global_step, loss.item(), CHECKPOINT_DIR)
                        # Optional head2head evaluation trigger
                        try:
                            eval_every = getattr(args, 'eval_every_steps', 0)
                            if eval_every and eval_every > 0 and (global_step % int(eval_every) == 0):
                                # Launch head2head asynchronously and forward its output
                                run_head2head_eval(ckpath, CHECKPOINT_DIR, getattr(args, 'eval_games', 50), getattr(args, 'eval_parallel', 8), getattr(args, 'eval_use_pytorch', False), getattr(args, 'eval_baseline', None))
                        except Exception:
                            pass
            # Step scheduler after optimizer update
            try:
                if 'scheduler' in locals() and scheduler is not None:
                    scheduler.step()
            except Exception:
                pass

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
    total_minutes = (end_time - start_time) / 60.0
    print(f"Training finished in {total_minutes:.2f} minutes.")

    # Save Final Model
    save_checkpoint(model, optimizer, EPOCHS, global_step, val_loss, CHECKPOINT_DIR)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/transformer_training_data_dummy.npz", help="Path to training data .npz file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (expand 8->16->32->64)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default="logs/transformer", help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/transformer", help="Directory for checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="Steps between checkpoints")
    # head2head evaluation options
    parser.add_argument("--eval_every_steps", type=int, default=0, help="0=disabled; run head2head every N steps after checkpoint")
    parser.add_argument("--eval_games", type=int, default=50, help="Number of games for head2head evaluation")
    parser.add_argument("--eval_parallel", type=int, default=8, help="Parallel size for head2head evaluation")
    parser.add_argument("--eval_use_pytorch", action='store_true', help="Load PyTorch .pth checkpoints for head2head evaluation")
    parser.add_argument("--eval_baseline", type=str, default="", help="Baseline model path for head2head evaluation (optional)")
    # AMP / accumulation / scheduler options
    parser.add_argument("--use_amp", action='store_true', help="Enable mixed precision (AMP) when CUDA available")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps to simulate larger effective batch size")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for LR scheduling")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="LR scheduler type: cosine|step")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Data file not found: {args.data_path}")
        print("Please run generate_transformer_training_data.py first.")
        exit(1)

    train(args)
    
