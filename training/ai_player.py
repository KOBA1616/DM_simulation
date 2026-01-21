import torch
import numpy as np
import os
import sys

# Try to import dm_toolkit
try:
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer
except ImportError:
    # Add root to path if needed
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dm_toolkit.ai.agent.transformer_model import DuelTransformer

try:
    import dm_ai_module
except ImportError:
    import dm_ai_module

class AIPlayer:
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        # Use same params as train_simple.py
        self.model = DuelTransformer(
            vocab_size=1000,
            action_dim=600,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            max_len=200
        ).to(device)

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                print(f"AIPlayer: Loaded model from {model_path}")
            except Exception as e:
                print(f"AIPlayer: Failed to load model: {e}")
        else:
            print(f"AIPlayer: Model not found at {model_path}, using random weights")

    def get_action(self, game_state):
        # Mocking input state tokenization
        # In real system, we'd use a tokenizer to convert game_state to tokens
        # Here we just create a dummy input
        state_tokens = torch.zeros((1, 200), dtype=torch.long).to(self.device)
        state_tokens[0, 0] = 1 # START token or similar

        with torch.no_grad():
            padding_mask = (state_tokens == 0)
            policy_logits, value = self.model(state_tokens, padding_mask=padding_mask)

        # Get best action index
        action_idx = torch.argmax(policy_logits[0]).item()

        # Map index to GameCommand
        # 0: PASS
        # 1-10: MANA CHARGE (index in hand)
        # 11-20: PLAY CARD (index in hand)
        # etc.

        cmd = dm_ai_module.GameCommand()

        if action_idx == 0:
            cmd.type = dm_ai_module.ActionType.PASS
        elif 1 <= action_idx <= 10:
            cmd.type = dm_ai_module.ActionType.MANA_CHARGE
            # In real logic we need to find the card at this index
            cmd.source_instance_id = -1
        elif 11 <= action_idx <= 20:
             cmd.type = dm_ai_module.ActionType.PLAY_CARD
             cmd.source_instance_id = -1
        else:
            # Default to PASS if unsure
            cmd.type = dm_ai_module.ActionType.PASS

        return cmd, action_idx
