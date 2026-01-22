from typing import Any, List
import numpy as np
import dm_ai_module
from dm_ai_module import GameCommand, ActionType, CommandType

class StateTokenizer:
    def __init__(self, vocab_size: int = 1000, max_len: int = 200):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def encode_state(self, state: Any, player_id: int) -> np.ndarray:
        tokens = []

        try:
            p = state.players[player_id]
            opp = state.players[1 - player_id]

            # MARKER: HAND (1001) -> use lower IDs or special tokens within vocab
            # For this simple test, we just assume markers are within range if vocab is large enough.
            # But 1001 > 1000 (vocab_size), causing IndexError.
            # Let's use 100, 101, 102, 103 for markers if vocab is small.
            # Or better, clamp markers too.

            # Simple fix: Use 999 for all markers or special reserved IDs < vocab_size
            MARKER_HAND = min(1001, self.vocab_size - 1)
            MARKER_MANA = min(1002, self.vocab_size - 1)
            MARKER_BATTLE = min(1003, self.vocab_size - 1)
            MARKER_OPP = min(1004, self.vocab_size - 1)

            tokens.append(MARKER_HAND)
            for c in p.hand:
                cid = c.card_id if hasattr(c, 'card_id') else c
                tokens.append(min(cid, self.vocab_size-1))

            tokens.append(MARKER_MANA)
            for c in p.mana_zone:
                cid = c.card_id if hasattr(c, 'card_id') else c
                tokens.append(min(cid, self.vocab_size-1))

            tokens.append(MARKER_BATTLE)
            for c in p.battle_zone:
                cid = c.card_id if hasattr(c, 'card_id') else c
                tokens.append(min(cid, self.vocab_size-1))

            tokens.append(MARKER_OPP)
            for c in opp.battle_zone:
                cid = c.card_id if hasattr(c, 'card_id') else c
                tokens.append(min(cid, self.vocab_size-1))

        except Exception as e:
            # Fallback for robustness
            pass

        # Pad/Truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [0] * (self.max_len - len(tokens))

        return np.array(tokens, dtype=np.int64)

class ActionEncoder:
    def __init__(self, action_dim: int = 600):
        self.action_dim = action_dim

    def decode_action(self, action_idx: int, state: Any, player_id: int) -> GameCommand:
        """
        Maps an integer index back to a GameCommand.
        Mapping logic (simplified for stub):
        0: PASS
        1..100: PLAY_CARD (index corresponds to hand index or known card ID map)
        101..200: MANA_CHARGE (hand index)
        201..300: ATTACK_PLAYER (battle zone index)
        301..400: ATTACK_CREATURE (battle zone index -> target index) - complex, simplified here
        """
        cmd = GameCommand()
        cmd.target_player = player_id

        # 0: PASS
        if action_idx == 0:
            cmd.type = ActionType.PASS
            return cmd

        # 1..10: MANA_CHARGE (Hand Index 0-9)
        if 1 <= action_idx <= 10:
            hand_idx = action_idx - 1
            p = state.players[player_id]
            if hand_idx < len(p.hand):
                c = p.hand[hand_idx]
                cmd.type = ActionType.MANA_CHARGE
                cmd.card_id = c.card_id
                cmd.source_instance_id = c.instance_id
                return cmd
            else:
                # Invalid index fallbacks to PASS
                cmd.type = ActionType.PASS
                return cmd

        # 11..20: PLAY_CARD (Hand Index 0-9)
        if 11 <= action_idx <= 20:
            hand_idx = action_idx - 11
            p = state.players[player_id]
            if hand_idx < len(p.hand):
                c = p.hand[hand_idx]
                cmd.type = ActionType.PLAY_CARD
                cmd.card_id = c.card_id
                cmd.source_instance_id = c.instance_id
                return cmd
            else:
                cmd.type = ActionType.PASS
                return cmd

        # 21..30: ATTACK_PLAYER (Battle Zone Index 0-9)
        if 21 <= action_idx <= 30:
            bz_idx = action_idx - 21
            p = state.players[player_id]
            if bz_idx < len(p.battle_zone):
                c = p.battle_zone[bz_idx]
                cmd.type = ActionType.ATTACK_PLAYER
                cmd.source_instance_id = c.instance_id
                cmd.target_player = 1 - player_id # Opponent
                return cmd
            else:
                cmd.type = ActionType.PASS
                return cmd

        # Default fallback
        cmd.type = ActionType.PASS
        return cmd

    def encode_action(self, action: GameCommand, state: Any, player_id: int) -> int:
        """
        Maps a GameCommand to an integer index.
        Returns -1 if the command cannot be encoded (invalid or out of range).
        """
        cmd_type = action.type

        # Check PASS
        # Handle both IntEnum and int comparison
        if cmd_type == ActionType.PASS or cmd_type == int(ActionType.PASS):
            return 0

        # Check MANA_CHARGE (1-10)
        if cmd_type == ActionType.MANA_CHARGE or cmd_type == int(ActionType.MANA_CHARGE):
            p = state.players[player_id]
            for i, c in enumerate(p.hand):
                if i >= 10: break
                # Check instance_id match
                # Use getattr for robustness across C++ objects and Python stubs
                c_id = getattr(c, 'instance_id', -1)
                cmd_id = getattr(action, 'source_instance_id', -2)
                if c_id == cmd_id:
                    return 1 + i
            return -1

        # Check PLAY_CARD (11-20)
        if cmd_type == ActionType.PLAY_CARD or cmd_type == int(ActionType.PLAY_CARD):
            p = state.players[player_id]
            for i, c in enumerate(p.hand):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                cmd_id = getattr(action, 'source_instance_id', -2)
                if c_id == cmd_id:
                    return 11 + i
            return -1

        # Check ATTACK_PLAYER (21-30)
        if cmd_type == ActionType.ATTACK_PLAYER or cmd_type == int(ActionType.ATTACK_PLAYER):
            p = state.players[player_id]
            for i, c in enumerate(p.battle_zone):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                cmd_id = getattr(action, 'source_instance_id', -2)
                if c_id == cmd_id:
                    return 21 + i
            return -1

        return -1
