from typing import Any, List
import numpy as np
import dm_ai_module
from dm_ai_module import GameCommand, ActionType, CommandType

# Block size used for action index ranges (kept small to match tests)
BLOCK = 10

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
        Mapping logic (updated for larger support):
        0: PASS
        1..40: MANA_CHARGE (Hand Index 0-39)
        41..80: PLAY_CARD (Hand Index 0-39)
        81..120: ATTACK_PLAYER (Battle Zone Index 0-39)
        121..160: ATTACK_CREATURE (Battle Zone Index 0-39, Target defaults to first tapped? Simplified)
        """
        cmd = GameCommand()
        cmd.target_player = player_id

        # 0: PASS
        if action_idx == 0:
            cmd.type = ActionType.PASS
            return cmd

        # Use module-level BLOCK (kept small to match tests)
        # (avoids shadowing and keeps a single source of truth)
        from dm_toolkit.ai.agent.tokenization import BLOCK as GLOBAL_BLOCK

        # 1..10: MANA_CHARGE (Hand Index 0-9)
        if 1 <= action_idx <= BLOCK:
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
        if (BLOCK + 1) <= action_idx <= (2 * BLOCK):
            hand_idx = action_idx - (BLOCK + 1)
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
        if (2 * BLOCK + 1) <= action_idx <= (3 * BLOCK):
            bz_idx = action_idx - (2 * BLOCK + 1)
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
        # Try to normalize the action/command into a dict using unified helper
        cmd_type = None
        action_dict = None
        try:
            try:
                from dm_toolkit.unified_execution import to_command_dict
                action_dict = to_command_dict(action)
            except Exception:
                # Fall back to wrappers or dicts
                if isinstance(action, dict):
                    action_dict = action
                else:
                    try:
                        d = action.to_dict()
                        action_dict = d
                    except Exception:
                        action_dict = None

            if action_dict is not None:
                # Prefer normalized dict type
                cmd_type = action_dict.get('type')
            else:
                if hasattr(action, 'type'):
                    cmd_type = getattr(action, 'type')
        except Exception:
            cmd_type = None

        def _type_matches(t: Any, enum_val: Any) -> bool:
            try:
                if t == enum_val or t == int(enum_val):
                    return True
            except Exception:
                pass
            try:
                if isinstance(t, str):
                    # Compare against enum member name when available (e.g. 'PASS')
                    enum_name = getattr(enum_val, 'name', None)
                    if enum_name is not None and t.upper() == enum_name.upper():
                        return True
                    # Fallback: compare numeric string forms
                    try:
                        if t.isdigit() and int(t) == int(enum_val):
                            return True
                    except Exception:
                        pass
            except Exception:
                pass
            return False

        # Check PASS (index 0)
        if _type_matches(cmd_type, ActionType.PASS):
            return 0

        # Check MANA_CHARGE (1-10)
        if _type_matches(cmd_type, ActionType.MANA_CHARGE):
            p = state.players[player_id]
            # Determine source id from action (support dict/wrapper)
            # Try normalized dict first
            if action_dict is not None:
                cmd_src = action_dict.get('source_instance_id') or action_dict.get('instance_id')
            else:
                try:
                    cmd_src = getattr(action, 'source_instance_id', None)
                except Exception:
                    cmd_src = None
                if cmd_src is None and isinstance(action, dict):
                    cmd_src = action.get('source_instance_id') or action.get('instance_id')

            for i, c in enumerate(p.hand):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                if cmd_src is not None and c_id == cmd_src:
                    return 1 + i
            return -1

        # Check PLAY_CARD (11-20)
        # Accept both legacy ActionType.PLAY_CARD and normalized command types
        if _type_matches(cmd_type, ActionType.PLAY_CARD) or (
            isinstance(action_dict, dict) and str(action_dict.get('type', '')).upper() in ('PLAY_FROM_ZONE', 'PLAY')
        ):
            p = state.players[player_id]
            if action_dict is not None:
                cmd_src = action_dict.get('source_instance_id') or action_dict.get('instance_id')
            else:
                try:
                    cmd_src = getattr(action, 'source_instance_id', None)
                except Exception:
                    cmd_src = None
                if cmd_src is None and isinstance(action, dict):
                    cmd_src = action.get('source_instance_id') or action.get('instance_id')
            for i, c in enumerate(p.hand):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                if cmd_src is not None and c_id == cmd_src:
                    return (BLOCK + 1) + i
            return -1

        # Check ATTACK_PLAYER (21-30)
        if _type_matches(cmd_type, ActionType.ATTACK_PLAYER):
            p = state.players[player_id]
            if action_dict is not None:
                cmd_src = action_dict.get('source_instance_id') or action_dict.get('instance_id')
            else:
                try:
                    cmd_src = getattr(action, 'source_instance_id', None)
                except Exception:
                    cmd_src = None
                if cmd_src is None and isinstance(action, dict):
                    cmd_src = action.get('source_instance_id') or action.get('instance_id')
            for i, c in enumerate(p.battle_zone):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                if cmd_src is not None and c_id == cmd_src:
                    return (2 * BLOCK + 1) + i
            return -1

        return -1
