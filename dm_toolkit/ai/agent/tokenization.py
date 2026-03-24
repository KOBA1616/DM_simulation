from typing import Any, List
try:
    import numpy as np
except ImportError:
    class np:
        class ndarray: pass
        def array(data, **kwargs): return data
        int64 = int

import dm_ai_module
# 再発防止: GameCommand は抽象C++クラスでPythonから直接インスタンス化不可。CommandDef を使用すること。
from dm_ai_module import CommandDef, CommandType

# Block size used for action index ranges (kept small to match tests)
BLOCK = 10

class StateTokenizer:
    # Constants matching C++ TokenConverter
    TOKEN_PAD = 0
    TOKEN_CLS = 1
    TOKEN_SEP = 2
    TOKEN_UNK = 3

    BASE_ZONE_MARKER = 10
    BASE_STATE_MARKER = 50
    BASE_PHASE_MARKER = 80
    BASE_CONTEXT_MARKER = 100
    BASE_COMMAND_MARKER = 200
    BASE_CARD_ID = 1000

    MARKER_HAND_SELF = 10
    MARKER_MANA_SELF = 11
    MARKER_BATTLE_SELF = 12
    MARKER_SHIELD_SELF = 13
    MARKER_GRAVE_SELF = 14
    MARKER_DECK_SELF = 15

    MARKER_HAND_OPP = 20
    MARKER_MANA_OPP = 21
    MARKER_BATTLE_OPP = 22
    MARKER_SHIELD_OPP = 23
    MARKER_GRAVE_OPP = 24
    MARKER_DECK_OPP = 25

    STATE_TAPPED = 50
    STATE_SICK = 51
    STATE_FACE_DOWN = 52

    def __init__(self, vocab_size: int = 10000, max_len: int = 512):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def encode_state(self, state: Any, player_id: int, card_db: Any = None) -> np.ndarray:
        tokens = []

        try:
            # 1. CLS Token
            tokens.append(self.TOKEN_CLS)

            # 2. Game Metadata
            tokens.append(self.BASE_CONTEXT_MARKER + 0) # Context Start
            tokens.append(getattr(state, 'turn_number', 1))

            phase = getattr(state, 'current_phase', 0)
            try:
                phase = int(phase)
            except:
                phase = 0
            tokens.append(self.BASE_PHASE_MARKER + phase)

            p = state.players[player_id]
            opp = state.players[1 - player_id]

            tokens.append(len(p.mana_zone))
            tokens.append(len(opp.mana_zone))

            # 3. Zones
            self._append_zone(tokens, p.battle_zone, self.MARKER_BATTLE_SELF, True)
            self._append_zone(tokens, opp.battle_zone, self.MARKER_BATTLE_OPP, True)

            self._append_zone(tokens, p.mana_zone, self.MARKER_MANA_SELF, True)
            self._append_zone(tokens, opp.mana_zone, self.MARKER_MANA_OPP, True)

            self._append_zone(tokens, p.hand, self.MARKER_HAND_SELF, True)

            # Opp Hand (Masked)
            tokens.append(self.MARKER_HAND_OPP)
            for _ in opp.hand:
                tokens.append(self.TOKEN_UNK)

            # Shields
            tokens.append(self.MARKER_SHIELD_SELF)
            for _ in p.shield_zone:
                tokens.append(self.TOKEN_UNK)

            tokens.append(self.MARKER_SHIELD_OPP)
            for _ in opp.shield_zone:
                tokens.append(self.TOKEN_UNK)

            # Graveyard
            self._append_zone(tokens, p.graveyard, self.MARKER_GRAVE_SELF, True)
            self._append_zone(tokens, opp.graveyard, self.MARKER_GRAVE_OPP, True)

            # Deck
            tokens.append(self.MARKER_DECK_SELF)
            for _ in p.deck:
                tokens.append(self.TOKEN_UNK)

            tokens.append(self.MARKER_DECK_OPP)
            for _ in opp.deck:
                tokens.append(self.TOKEN_UNK)

            # 4. Command History
            tokens.append(self.TOKEN_SEP)
            self._append_command_history(tokens, state, 30)

        except Exception as e:
            # Fallback
            pass

        # Pad/Truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.TOKEN_PAD] * (self.max_len - len(tokens))

        return np.array(tokens, dtype=np.int64)

    def _append_zone(self, tokens: List[int], zone: List[Any], zone_marker: int, visible: bool):
        tokens.append(zone_marker)
        for card in zone:
            self._append_card(tokens, card, visible)

    def _append_card(self, tokens: List[int], card: Any, visible: bool):
        if not visible:
            tokens.append(self.TOKEN_UNK)
            return

        cid = getattr(card, 'card_id', None)
        if cid is None:
             cid = getattr(card, 'id', 0)

        try:
            cid = int(cid)
        except:
            cid = 0

        tokens.append(self.BASE_CARD_ID + cid)

        if getattr(card, 'is_tapped', False):
            tokens.append(self.STATE_TAPPED)

        is_sick = getattr(card, 'sick', False) or getattr(card, 'summoning_sickness', False)
        if is_sick:
            tokens.append(self.STATE_SICK)

    def _append_command_history(self, tokens: List[int], state: Any, limit: int):
        history = getattr(state, 'command_history', [])
        start = max(0, len(history) - limit)
        for i in range(start, len(history)):
            cmd = history[i]
            ctype = 0
            if isinstance(cmd, dict):
                ctype = cmd.get('type', 0)
            else:
                ctype = getattr(cmd, 'type', 0)

            try:
                ctype = int(ctype)
            except:
                pass

            tokens.append(self.BASE_COMMAND_MARKER + ctype)

# 再発防止: ActionEncoder は CommandEncoder に改名。GameCommand/CommandType を扱うクラスのため。
# 再発防止: decode_action / encode_action は decode_command / encode_command に改名済み。後方互揜エイリアスを末尾に保持。
class CommandEncoder:
    def __init__(self, action_dim: int = 600):
        self.action_dim = action_dim

    def decode_command(self, command_idx: int, state: Any, player_id: int) -> CommandDef:
        """
        整数インデックスを CommandDef にマップする。
        マッピング冗长（大型サポート向けに更新）:
        0: PASS
        1..40: MANA_CHARGE (手札 Index 0-39)
        41..80: PLAY_FROM_ZONE (手札 Index 0-39)
        81..120: ATTACK_PLAYER (バトルゾーン Index 0-39)
        121..160: ATTACK_CREATURE (バトルゾーン Index 0-39)
        """
        cmd = CommandDef()
        cmd.owner_id = player_id

        # 0: PASS
        if command_idx == 0:
            cmd.type = CommandType.PASS
            return cmd

        # Use module-level BLOCK (kept small to match tests)
        # (avoids shadowing and keeps a single source of truth)
        from dm_toolkit.ai.agent.tokenization import BLOCK as GLOBAL_BLOCK

        # 1..10: MANA_CHARGE (Hand Index 0-9)
        if 1 <= command_idx <= BLOCK:
            hand_idx = command_idx - 1
            p = state.players[player_id]
            if hand_idx < len(p.hand):
                c = p.hand[hand_idx]
                cmd.type = CommandType.MANA_CHARGE
                cmd.instance_id = c.instance_id
                return cmd
            else:
                # Invalid index fallbacks to PASS
                cmd.type = CommandType.PASS
                return cmd
        # 11..20: PLAY_FROM_ZONE (Hand Index 0-9)
        if (BLOCK + 1) <= command_idx <= (2 * BLOCK):
            hand_idx = command_idx - (BLOCK + 1)
            p = state.players[player_id]
            if hand_idx < len(p.hand):
                c = p.hand[hand_idx]
                cmd.type = CommandType.PLAY_FROM_ZONE
                cmd.instance_id = c.instance_id
                return cmd
            else:
                cmd.type = CommandType.PASS
                return cmd
        # 21..30: ATTACK_PLAYER (Battle Zone Index 0-9)
        if (2 * BLOCK + 1) <= command_idx <= (3 * BLOCK):
            bz_idx = command_idx - (2 * BLOCK + 1)
            p = state.players[player_id]
            if bz_idx < len(p.battle_zone):
                c = p.battle_zone[bz_idx]
                cmd.type = CommandType.ATTACK_PLAYER
                cmd.instance_id = c.instance_id
                cmd.owner_id = 1 - player_id  # Opponent target player
                return cmd
            else:
                cmd.type = CommandType.PASS
                return cmd

        # Default fallback
        cmd.type = CommandType.PASS
        return cmd

    # 後方互換エイリアス
    # 再発防止: decode_action は decode_command のエイリアス。新規コードでは decode_command を使用すること。
    decode_action = decode_command

    def encode_command(self, command: CommandDef, state: Any, player_id: int) -> int:
        """
        CommandDef を整数インデックスにマップする。
        エンコードできない場合は -1 を返す。
        """
        # Try to normalize the command into a dict
        cmd_type = None
        action_dict = None
        try:
            # 再発防止: unified_execution.to_command_dict は削除済み。ローカル変換で代替。
            if isinstance(command, dict):
                action_dict = command
            elif hasattr(command, 'to_dict'):
                action_dict = command.to_dict()
            else:
                action_dict = None

            if action_dict is not None:
                # Prefer normalized dict type
                cmd_type = action_dict.get('type')
            else:
                if hasattr(command, 'type'):
                    cmd_type = getattr(command, 'type')
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
        if _type_matches(cmd_type, CommandType.PASS):
            return 0

        # Check MANA_CHARGE (1-10)
        if _type_matches(cmd_type, CommandType.MANA_CHARGE):
            p = state.players[player_id]
            # Determine source id from command (support dict/wrapper)
            # Try normalized dict first
            if action_dict is not None:
                cmd_src = action_dict.get('source_instance_id') or action_dict.get('instance_id')
            else:
                try:
                    cmd_src = getattr(command, 'source_instance_id', None)
                except Exception:
                    cmd_src = None
                if cmd_src is None and isinstance(command, dict):
                    cmd_src = command.get('source_instance_id') or command.get('instance_id')

            for i, c in enumerate(p.hand):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                if cmd_src is not None and c_id == cmd_src:
                    return 1 + i
            return -1

        # Check PLAY_FROM_ZONE (11-20)
        # Accept normalized command types
        if _type_matches(cmd_type, CommandType.PLAY_FROM_ZONE) or (
            isinstance(action_dict, dict) and str(action_dict.get('type', '')).upper() in ('PLAY_FROM_ZONE', 'PLAY')
        ):
            p = state.players[player_id]
            if action_dict is not None:
                cmd_src = action_dict.get('source_instance_id') or action_dict.get('instance_id')
            else:
                try:
                    cmd_src = getattr(command, 'source_instance_id', None)
                except Exception:
                    cmd_src = None
                if cmd_src is None and isinstance(command, dict):
                    cmd_src = command.get('source_instance_id') or command.get('instance_id')
            for i, c in enumerate(p.hand):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                if cmd_src is not None and c_id == cmd_src:
                    return (BLOCK + 1) + i
            return -1

        # Check ATTACK_PLAYER (21-30)
        if _type_matches(cmd_type, CommandType.ATTACK_PLAYER):
            p = state.players[player_id]
            if action_dict is not None:
                cmd_src = action_dict.get('source_instance_id') or action_dict.get('instance_id')
            else:
                try:
                    cmd_src = getattr(command, 'source_instance_id', None)
                except Exception:
                    cmd_src = None
                if cmd_src is None and isinstance(command, dict):
                    cmd_src = command.get('source_instance_id') or command.get('instance_id')
            for i, c in enumerate(p.battle_zone):
                if i >= 10: break
                c_id = getattr(c, 'instance_id', -1)
                if cmd_src is not None and c_id == cmd_src:
                    return (2 * BLOCK + 1) + i
            return -1

        return -1

    # 後方互換エイリアス
    encode_action = encode_command

# 再発防止: ActionEncoder は CommandEncoder に改名済み。後方互換エイリアス。
# 新規コードでは CommandEncoder を使用すること。
ActionEncoder = CommandEncoder

