
import torch
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.tokenization import StateTokenizer, ActionEncoder
from dm_ai_module import GameCommand

class AIPlayer:
    def __init__(self, model_path: str, device='cpu', config=None):
        self.device = device
        self.tokenizer = StateTokenizer()
        self.action_encoder = ActionEncoder()

        # Default Config (matches train_simple.py)
        self.config = config or {
            'vocab_size': 1000,
            'action_dim': 600,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'max_len': 200
        }

        self.model = DuelTransformer(
            vocab_size=self.config['vocab_size'],
            action_dim=self.config['action_dim'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            max_len=self.config['max_len']
        ).to(self.device)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
            except RuntimeError as e:
                print(f"Warning: Model architecture mismatch. {e}")
                print("Proceeding with initialized weights (random).")
        else:
            print(f"Warning: Model not found at {model_path}. Using random weights.")

        self.model.eval()

    def get_action(self, game_state, player_id: int, valid_indices: list[int] = None) -> GameCommand:
        # 1. Tokenize
        state_tokens = self.tokenizer.encode_state(game_state, player_id)

        state_tensor = torch.tensor([state_tokens], dtype=torch.long).to(self.device)
        padding_mask = (state_tensor == 0)

        # 2. Inference
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor, padding_mask=padding_mask)

        # 3. Masking (Optional)
        if valid_indices is not None and len(valid_indices) > 0:
            # Create a mask with -inf for invalid actions
            full_mask = torch.full_like(policy_logits, float('-inf'))

            # Filter valid indices that are within range
            safe_indices = [idx for idx in valid_indices if 0 <= idx < policy_logits.shape[1]]

            if safe_indices:
                full_mask[0, safe_indices] = 0
                # Apply mask
                policy_logits = policy_logits + full_mask
            else:
                pass # No valid indices in range? Fallback to raw model output

        # 4. Decode
        action_idx = torch.argmax(policy_logits, dim=1).item()

        # 5. Map to GameCommand
        command = self.action_encoder.decode_action(action_idx, game_state, player_id)
        return command
