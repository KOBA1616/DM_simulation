import torch
import dm_ai_module
from dm_toolkit.ai.agent.transformer_model import DuelTransformer

class SimpleActionDecoder:
    """
    Decodes integer action ID into a GameCommand.
    Corresponds to the action encoding used in training/simple_game_generator.py
    """
    def decode(self, action_idx: int, game_state: 'dm_ai_module.GameState') -> 'dm_ai_module.GameCommand':
        # Simple mapping for Phase 1
        # 0 -> PASS
        # 1..10 -> MANA_CHARGE Hand Card [0..9]
        # 11..20 -> ATTACK_PLAYER with Battle Zone Card [0..9]
        # This is a temporary mapping until the C++ ActionEncoder is exposed or replicated

        cmd = dm_ai_module.GameCommand()

        if action_idx == 0:
            cmd.type = dm_ai_module.ActionType.PASS

        elif 1 <= action_idx <= 10:
            # Map to Mana Charge for testing
            hand_idx = action_idx - 1
            player = game_state.players[game_state.active_player_id]
            if hand_idx < len(player.hand):
                card = player.hand[hand_idx]
                cmd.type = dm_ai_module.ActionType.MANA_CHARGE
                cmd.source_instance_id = card.instance_id
            else:
                cmd.type = dm_ai_module.ActionType.PASS

        elif 11 <= action_idx <= 20:
            # Map to Attack Player
            creature_idx = action_idx - 11
            player = game_state.players[game_state.active_player_id]
            if creature_idx < len(player.battle_zone):
                card = player.battle_zone[creature_idx]
                cmd.type = dm_ai_module.ActionType.ATTACK_PLAYER
                cmd.source_instance_id = card.instance_id
                cmd.target_player = 1 - game_state.active_player_id
            else:
                cmd.type = dm_ai_module.ActionType.PASS

        else:
             cmd.type = dm_ai_module.ActionType.PASS

        return cmd

class AIPlayer:
    """
    AI Player that uses a trained DuelTransformer model to decide actions.
    """
    def __init__(self, model_path: str, device='cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.decoder = SimpleActionDecoder()

    def _load_model(self, path):
        # Initialize model with same params as training
        # Must match training/train_simple.py
        model = DuelTransformer(
            vocab_size=1000,
            action_dim=591,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            max_len=200
        )

        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {path}")
        except FileNotFoundError:
            print(f"Warning: Model file {path} not found. Using random weights for testing.")
        except Exception as e:
            print(f"Warning: Failed to load model: {e}. Using random weights.")

        model.to(self.device)
        model.eval()
        return model

    def encode_state(self, gs) -> torch.Tensor:
        # Replicate SimpleGameGenerator._encode_state
        tokens = [0] * 200
        if len(gs.players) > 0:
            active_p = gs.active_player_id
            # Safety check for minimal stubs
            p0_shields = len(gs.players[0].shield_zone) if hasattr(gs.players[0], 'shield_zone') else 0
            p1_shields = len(gs.players[1].shield_zone) if hasattr(gs.players[1], 'shield_zone') else 0

            my_shields = p0_shields if active_p == 0 else p1_shields
            opp_shields = p1_shields if active_p == 0 else p0_shields

            tokens[0] = min(my_shields, 10)
            tokens[1] = min(opp_shields, 10)
            tokens[2] = active_p + 1
            tokens[3] = gs.turn_number if hasattr(gs, 'turn_number') else 1

        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def get_action(self, game_state) -> 'dm_ai_module.GameCommand':
        state_tensor = self.encode_state(game_state)

        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            # Simple greedy sampling
            action_idx = torch.argmax(policy_logits, dim=1).item()

        return self.decoder.decode(action_idx, game_state)
