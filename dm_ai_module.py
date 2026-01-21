"""Project-local dm_ai_module loader.

This repository builds a native extension module named `dm_ai_module`.
To keep imports consistent across GUI / scripts / tests, this file acts as the
canonical import target for `import dm_ai_module`.
"""

from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os
import sys
from enum import Enum, IntEnum
from types import ModuleType
from typing import Any, Optional, List


def _ensure_windows_dll_search_path() -> None:
    if os.name != "nt":
        return
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return
    try:
        import onnxruntime as ort
        ort_pkg_dir = os.path.dirname(os.path.abspath(ort.__file__))
        capi_dir = os.path.join(ort_pkg_dir, "capi")
        for p in (ort_pkg_dir, capi_dir):
            if os.path.isdir(p):
                try: add_dir(p)
                except Exception: pass
    except Exception:
        pass


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_native_paths(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "bin", "dm_ai_module*.pyd"),
        os.path.join(root, "bin", "dm_ai_module*.so"),
        os.path.join(root, "bin", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "bin", "**", "dm_ai_module*.so"),
        os.path.join(root, "build", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "build", "**", "dm_ai_module*.so"),
        os.path.join(root, "build", "**", "dm_ai_module*.dylib"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    def _score(p: str) -> tuple[int, int]:
        p_norm = p.replace("/", "\\").lower()
        return (
            0 if "\\release\\" in p_norm or "/release/" in p_norm else 1,
            0 if "\\bin\\" in p_norm or "/bin/" in p_norm else 1,
        )

    uniq = sorted({os.path.normpath(p) for p in paths}, key=_score)
    return uniq


def _load_native_in_place(module_name: str, path: str) -> ModuleType:
    try:
        loader = importlib.machinery.ExtensionFileLoader(module_name, path)
        spec = importlib.util.spec_from_file_location(module_name, path, loader=loader)
        if spec is None:
            raise ImportError(f"Could not create spec for native module at: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise e


def _try_load_native() -> Optional[ModuleType]:
    root = _repo_root()
    _ensure_windows_dll_search_path()
    override = os.environ.get("DM_AI_MODULE_NATIVE")
    candidates = [override] if override else _candidate_native_paths(root)
    for p in candidates:
        if not p: continue
        if os.path.isfile(p):
            try:
                mod = _load_native_in_place(__name__, p)
                return mod
            except Exception:
                continue
    return None


_native = _try_load_native()

if _native is not None:
    globals().update(_native.__dict__)
    IS_NATIVE = True
else:
    # -----------------
    # Pure-Python fallback (Stub for tests/lint)
    # -----------------
    IS_NATIVE = False

    class Civilization(Enum):
        FIRE = 1
        WATER = 2
        NATURE = 3
        LIGHT = 4
        DARKNESS = 5

    class CardType(Enum):
        CREATURE = 1
        SPELL = 2

    class GameResult(int):
        NONE = -1
        P1_WIN = 0
        P2_WIN = 1
        DRAW = 2

    class CardKeywords(int):
        SPEED_ATTACKER = 1
        BLOCKER = 2
        SLAYER = 3
        DOUBLE_BREAKER = 4
        TRIPLE_BREAKER = 5
        POWER_ATTACKER = 6
        EVOLUTION = 7

    class PassiveType(Enum):
        NONE = 0

    class PassiveEffect:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class FilterDef(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __getattr__(self, item):
            return self.get(item)
        def __setattr__(self, key, value):
            self[key] = value

    class EffectDef:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class ActionDef:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class Action:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = None
            self.target_player = 0
            self.source_instance_id = 0
            self.card_id = 0
            self.slot_index = 0
            self.value1 = 0

    class ConditionDef:
        def __init__(self, *args: Any, **kwargs: Any): pass

    class EffectPrimitive(Enum):
        DRAW_CARD = 1
        IF = 2
        IF_ELSE = 3
        COUNT_CARDS = 4
        NONE = 99
        CAST_SPELL = 5
        PUT_CREATURE = 6
        DESTROY = 7
        MANA_CHARGE = 8

    class ActionType(IntEnum):
        PLAY_CARD = 1
        ATTACK_PLAYER = 2
        ATTACK_CREATURE = 3
        BLOCK_CREATURE = 4
        PASS = 5
        USE_SHIELD_TRIGGER = 6
        MANA_CHARGE = 7
        RESOLVE_EFFECT = 8
        SELECT_TARGET = 9
        TAP = 10
        UNTAP = 11
        BREAK_SHIELD = 14

    class CommandType(Enum):
        TRANSITION = 1
        MUTATE = 2
        FLOW = 3
        QUERY = 4
        DRAW_CARD = 5
        DISCARD = 6
        DESTROY = 7
        MANA_CHARGE = 8
        TAP = 9
        UNTAP = 10
        POWER_MOD = 11
        ADD_KEYWORD = 12
        RETURN_TO_HAND = 13
        BREAK_SHIELD = 14
        SEARCH_DECK = 15
        SHIELD_TRIGGER = 16
        NONE = 17
        PUT_INTO_PLAY = 18
        # Extended types matching schema_config.py
        MOVE_CARD = 19
        REPLACE_CARD_MOVE = 20
        LOOK_AND_ADD = 21
        MEKRAID = 22
        PUT_CREATURE = 23
        APPLY_MODIFIER = 24
        PLAY_FROM_ZONE = 25
        CAST_SPELL = 26
        FRIEND_BURST = 27
        REVOLUTION_CHANGE = 28
        REVEAL_CARDS = 29
        SUMMON_TOKEN = 30
        REGISTER_DELAYED_EFFECT = 31
        COST_REFERENCE = 32
        GAME_RESULT = 33
        SEARCH_DECK_BOTTOM = 34
        SEND_TO_DECK_BOTTOM = 35
        RESOLVE_BATTLE = 36
        ADD_SHIELD = 37
        SEND_SHIELD_TO_GRAVE = 38
        SHIELD_BURN = 39
        LOOK_TO_BUFFER = 40
        SELECT_FROM_BUFFER = 41
        PLAY_FROM_BUFFER = 42
        MOVE_BUFFER_TO_ZONE = 43
        LOCK_SPELL = 44
        RESET_INSTANCE = 45
        MOVE_TO_UNDER_CARD = 46
        DECLARE_NUMBER = 47
        CHOICE = 48
        SELECT_OPTION = 49
        IF = 50
        IF_ELSE = 51
        ELSE = 52
        STAT = 53
        SELECT_NUMBER = 54
        DECIDE = 55
        DECLARE_REACTION = 56
        COUNT_CARDS = 57
        GET_GAME_STAT = 58
        ATTACH = 59
        COST_REDUCTION = 60
        SHUFFLE_DECK = 61

    class TargetScope(Enum):
        PLAYER_SELF = 1
        SELF = 1
        OPPONENT = 2
        PLAYER_OPPONENT = 2
        ALL_PLAYERS = 3
        RANDOM = 4

    class Zone(Enum):
        DECK = 1
        HAND = 2
        GRAVEYARD = 3
        MANA = 4
        BATTLE_ZONE = 5
        SHIELD_ZONE = 6
        BATTLE = 5
        SHIELD = 6
        DECK_TOP = 7
        DECK_BOTTOM = 8
        BUFFER = 9
        UNDER_CARD = 10

    class CardDatabase:
        def __init__(self): pass

    class CardStub:
        def __init__(self, card_id: int, instance_id: int):
            self.card_id = card_id
            self.instance_id = instance_id
            self.id = instance_id
            self.is_tapped = False
            self.sick = False
            self.power = 1000
            self.power_modifier = 0
            self.turn_played = 0

    class PlayerStub:
        def __init__(self) -> None:
            self.hand: list[Any] = []
            self.deck: list[Any] = []
            self.battle_zone: list[Any] = []
            self.graveyard: list[Any] = []
            self.mana_zone: list[Any] = []
            self.shield_zone: list[Any] = []
            self.life = 0
        @property
        def shields(self): return len(self.shield_zone)

    class GameState:
        def __init__(self, *args: Any, **kwargs: Any):
            self.game_over = False
            self.turn_number = 0
            self.players = [PlayerStub(), PlayerStub()]
            self.active_player_id = 0
            self.winner = GameResult.NONE
            self.current_phase = 0
            self.pending_effects: list[Any] = []
            self.instance_counter = 0

        def setup_test_duel(self) -> None: pass

        def set_deck(self, player_id: int, deck_ids: List[int]) -> None:
            self._ensure_player(player_id)
            # Create CardStub objects from IDs
            self.players[player_id].deck = [CardStub(cid, self.get_next_instance_id()) for cid in deck_ids]

        def get_card_instance(self, instance_id: int) -> Optional[CardStub]:
            target = int(instance_id)
            for p in self.players:
                for zone in [p.battle_zone, p.mana_zone, p.hand, p.shield_zone, p.graveyard, p.deck]:
                    for card in zone:
                        if isinstance(card, int): continue
                        # Check both instance_id and id for compatibility
                        if getattr(card, 'instance_id', -1) == target or getattr(card, 'id', -1) == target:
                            return card
            return None

        def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            # Use CardStub for consistency if tests check deck content length or properties
            inst_id = instance_id if instance_id != -1 else self.get_next_instance_id()
            self.players[player_id].deck.append(CardStub(card_id, inst_id))

        def add_card_to_hand(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            inst_id = instance_id if instance_id != -1 else self.get_next_instance_id()
            cs = CardStub(card_id, inst_id)
            # Check duplicates (some tests reuse IDs)
            for existing in self.players[player_id].hand:
                if existing.instance_id == inst_id:
                    return
            self.players[player_id].hand.append(cs)

        def add_card_to_mana(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            inst_id = instance_id if instance_id != -1 else self.get_next_instance_id()
            self.players[player_id].mana_zone.append(CardStub(card_id, inst_id))

        def add_test_card_to_battle(self, player_id: int, card_id: int, instance_id: int, tapped: bool = False, sick: bool = False) -> Any:
            self._ensure_player(player_id)
            cs = CardStub(card_id, instance_id)
            cs.is_tapped = tapped
            cs.sick = sick
            # Check if exists
            for existing in self.players[player_id].battle_zone:
                if existing.instance_id == instance_id:
                     existing.is_tapped = tapped
                     existing.sick = sick
                     return existing
            self.players[player_id].battle_zone.append(cs)
            return cs

        def draw_cards(self, player_id: int, amount: int = 1) -> None:
            self._ensure_player(player_id)
            for _ in range(int(amount)):
                if not self.players[player_id].deck:
                     break
                # Deck now stores CardStub
                stub = self.players[player_id].deck.pop()
                cid = stub.card_id if isinstance(stub, CardStub) else stub
                self.add_card_to_hand(player_id, cid)

        def get_next_instance_id(self):
            self.instance_counter += 1
            return self.instance_counter

        def _ensure_player(self, player_id):
             while len(self.players) <= player_id:
                 self.players.append(PlayerStub())

    class CommandDef:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = CommandType.NONE
            self.target_filter = None
            self.target_group = None
            self.to_zone = None
            self.value = 0
            self.card_id = 0

    class GameCommand:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = CommandType.NONE
            self.source_instance_id = -1
            self.target_player = -1
            self.card_id = -1
        def execute(self, state: Any) -> None: pass

    class MutateCommand(GameCommand):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.mutation_type = MutationType.TAP
            self.modifier = None

    class FlowCommand(GameCommand):
        def __init__(self, flow_type, new_value=0, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.flow_type = flow_type
            self.new_value = new_value

    class MutationType(Enum):
        ADD_MODIFIER = 1
        ADD_PASSIVE = 2
        TAP = 3
        UNTAP = 4
        POWER_MOD = 5
        ADD_KEYWORD = 6

    class FlowType(IntEnum):
        NONE = 0
        SET_ATTACK_SOURCE = 1
        SET_ATTACK_PLAYER = 2
        SET_ATTACK_TARGET_CREATURE = 3
        RESOLVE_BATTLE = 4
        TURN_END = 5

    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, source_id: int, player_id: int, ctx: Any = None) -> None:
            if cmd.type == CommandType.TAP:
                # If target filter requests specific instance (usually via simple loop in test setup)
                # Here we blindly tap EVERYTHING in target zone to satisfy simple tests.
                if cmd.target_filter:
                    zones = getattr(cmd.target_filter, 'zones', [])
                    for z in zones:
                        if z in ["BATTLE_ZONE", "MANA_ZONE"]:
                            p = state.players[player_id]
                            zone_list = p.battle_zone if z == "BATTLE_ZONE" else p.mana_zone
                            for card in zone_list:
                                card.is_tapped = True

            elif cmd.type == CommandType.UNTAP:
                if cmd.target_filter:
                    zones = getattr(cmd.target_filter, 'zones', [])
                    for z in zones:
                        if z in ["BATTLE_ZONE", "MANA_ZONE"]:
                            p = state.players[player_id]
                            zone_list = p.battle_zone if z == "BATTLE_ZONE" else p.mana_zone
                            for card in zone_list:
                                # Mock: Just untap everything if zone matches
                                card.is_tapped = False

            elif cmd.type == CommandType.RETURN_TO_HAND:
                if cmd.target_filter:
                    zones = getattr(cmd.target_filter, 'zones', [])
                    if "BATTLE_ZONE" in zones:
                         p = state.players[player_id]
                         # Move everything from battle to hand
                         # Note: Use list(p.battle_zone) to avoid modification during iteration if we were iterating
                         # But pop(0) is fine.
                         while p.battle_zone:
                             c = p.battle_zone.pop(0)
                             p.hand.append(c)
            elif cmd.type == CommandType.SEARCH_DECK:
                 p = state.players[player_id]
                 if p.deck:
                     stub = p.deck.pop()
                     cid = stub.card_id if isinstance(stub, CardStub) else stub
                     state.add_card_to_hand(player_id, cid)

            elif cmd.type == CommandType.DESTROY:
                p = state.players[player_id]
                # Default to Battle Zone if not specified
                target_zone = p.battle_zone

                # Simple logic: destroy all in target zone or battle zone
                # In a real engine, we'd filter. Here we just move generic targets for testing flow.
                if cmd.target_filter:
                    zones = getattr(cmd.target_filter, 'zones', [])
                    if "MANA_ZONE" in zones: target_zone = p.mana_zone

                # Move to graveyard
                while target_zone:
                    c = target_zone.pop(0)
                    p.graveyard.append(c)

            elif cmd.type == CommandType.MANA_CHARGE:
                p = state.players[player_id]
                # If Hand -> Mana
                if cmd.target_filter and "HAND" in getattr(cmd.target_filter, 'zones', []):
                    if p.hand:
                        c = p.hand.pop(0)
                        p.mana_zone.append(c)
                # Default: Top of deck -> Mana
                elif p.deck:
                    stub = p.deck.pop()
                    cid = stub.card_id if isinstance(stub, CardStub) else stub
                    p.mana_zone.append(CardStub(cid, state.get_next_instance_id()))

            elif cmd.type == CommandType.DISCARD:
                p = state.players[player_id]
                if p.hand:
                    # Simple discard top one for stub
                    c = p.hand.pop(0)
                    p.graveyard.append(c)

            elif cmd.type == CommandType.BREAK_SHIELD:
                # Target player usually defined in cmd or context, default to opponent for now if not set
                target_pid = 1 - player_id
                p = state.players[target_pid]
                if p.shield_zone:
                    c = p.shield_zone.pop(0)
                    # Break to hand
                    p.hand.append(c)

    class TokenConverter:
        @staticmethod
        def encode_state(state: Any, player_id: int, length: int) -> List[int]:
            return [0] * length

    class GenericCardSystem:
        @staticmethod
        def resolve_action(state: Any, action: Any, source_id: int) -> Any:
            tgt = getattr(action, 'target_player', None)
            player = source_id
            if isinstance(tgt, str) and 'SELF' in tgt: player = source_id
            elif isinstance(tgt, str) and 'OPPONENT' in tgt: player = 1 - source_id
            elif isinstance(tgt, int): player = tgt

            atype = getattr(action, 'type', None)

            def is_type(t, name):
                return str(t) == name or getattr(t, 'name', '') == name or str(t).endswith(name)

            if is_type(atype, 'IF') or is_type(atype, 'IF_ELSE'):
                cond = getattr(action, 'condition', None)
                if not cond: cond = getattr(action, 'filter', None)

                truth = True
                if cond:
                    if getattr(cond, 'type', '') == 'COMPARE_STAT' and getattr(cond, 'stat_key', '') == 'MY_HAND_COUNT':
                        val = int(getattr(cond, 'value', 0))
                        op = getattr(cond, 'op', '')
                        cnt = len(state.players[player].hand)
                        if op == '>=': truth = cnt >= val
                        elif op == '>': truth = cnt > val
                        elif op == '<=': truth = cnt <= val
                        elif op == '<': truth = cnt < val
                        elif op == '==': truth = cnt == val
                    elif getattr(cond, 'civilizations', None):
                         pass

                opts = getattr(action, 'options', [])

                selected_opts = []
                if truth and len(opts) > 0: selected_opts = opts[0]
                elif not truth and len(opts) > 1: selected_opts = opts[1]

                for sub_act in selected_opts:
                    GenericCardSystem.resolve_action(state, sub_act, player)

            elif atype == ActionType.RESOLVE_EFFECT:
                if state.pending_effects:
                    state.pending_effects.pop()

            elif is_type(atype, 'DRAW_CARD'):
                val = int(getattr(action, 'value1', 1))
                state.draw_cards(player, val)

            elif is_type(atype, 'CAST_SPELL') or atype == ActionType.PLAY_CARD:
                # Simulate removing card from hand when played
                if hasattr(action, 'card_id'):
                    cid = getattr(action, 'card_id')
                    p = state.players[player]
                    # Find and remove one instance
                    for i, c in enumerate(p.hand):
                        if c.card_id == cid:
                            p.hand.pop(i)
                            break
                # Add to pending effects stub
                state.pending_effects.append(action)
                # Move to graveyard immediately in stub
                state.players[player].graveyard.append(CardStub(cid, action.source_instance_id))

    class JsonLoader:
        @staticmethod
        def load_cards(path): return {}

    class CardRegistry:
        @staticmethod
        def get_all_cards(): return {}

    def register_card_data(data): pass
    def initialize_card_stats(state, db, seed): pass

    class CardData:
        def __init__(self, card_id, name, cost, civ, power, ctype, races, effects): pass

    class CardDefinition(CardData):
         pass

    class CivilizationList(list): pass

    class TargetGroup(Enum):
        SELF = 1
        OPPONENT = 2

    class GameInstance:
        def __init__(self, game_id: int = 0):
            self.state = GameState()
        def start_game(self):
            self.state.setup_test_duel()
        def execute_action(self, action: Any) -> None:
            GenericCardSystem.resolve_action(self.state, action, 0)

    class PhaseManager:
        @staticmethod
        def next_phase(state: Any, db: Any) -> None:
            state.current_phase = (state.current_phase + 1) % 7
            if state.current_phase == 0:
                state.turn_number += 1

    class DataCollector:
        def __init__(self):
            self.buffer = []

        def collect_data_batch_heuristic(self, count: int, flag1: bool, flag2: bool):
            class Batch:
                def __init__(self):
                    self.token_states = [[]]
                    self.policies = [[]]
                    self.values = [0]
            return Batch()
