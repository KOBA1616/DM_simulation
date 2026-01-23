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
import weakref
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
        CANNOT_ATTACK = 1

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

        def __repr__(self):
            return f"<Action type={self.type} card={self.card_id} source={self.source_instance_id}>"

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

    class Phase(IntEnum):
        START = 0
        DRAW = 1
        MANA = 2
        MAIN = 3
        ATTACK = 4
        END = 5

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
        _cards = {}
        _loaded = False

        @staticmethod
        def load(path: str = "data/cards.json") -> None:
            if CardDatabase._loaded: return
            try:
                # Attempt to resolve path relative to repo root if not found
                if not os.path.exists(path):
                    root = _repo_root()
                    path = os.path.join(root, "data", "cards.json")

                if os.path.exists(path):
                    import json
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for c in data:
                            CardDatabase._cards[c['id']] = c
                    CardDatabase._loaded = True
            except Exception as e:
                print(f"Warning: Failed to load card database: {e}")

        @staticmethod
        def get_card(card_id: int) -> dict:
            if not CardDatabase._loaded:
                CardDatabase.load()
            return CardDatabase._cards.get(card_id, {})

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

    class ExecutionContext:
        def __init__(self):
            self.variables = {}

        def set_variable(self, name: str, value: Any):
            self.variables[name] = value

    class GameState:
        def __init__(self, *args: Any, **kwargs: Any):
            self.game_over = False
            # Default to turn 1 for tests expecting game start
            self.turn_number = 1
            self.players = [PlayerStub(), PlayerStub()]
            self.active_player_id = 0
            self.winner = GameResult.NONE
            self.current_phase = 0
            self.pending_effects: list[Any] = []
            self.instance_counter = 0
            self.execution_context = ExecutionContext()
            self.waiting_for_user_input = False
            self.pending_query = None
            self.command_history: list[Any] = []
            self.effect_buffer: list[Any] = []

        def setup_test_duel(self) -> None: pass

        def initialize(self) -> None: pass

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
            self.instance_id = 0
            self.target_instance = 0
            self.owner_id = 0
            self.from_zone = ''
            self.mutation_kind = ''
            self.input_value_key = ''
            self.output_value_key = ''

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

    class ActionGenerator:
        @staticmethod
        def generate_legal_actions(state: Any, card_db: Any) -> List[Any]:
            actions = []

            # Helper to get card data
            def get_card_def(cid):
                if hasattr(card_db, 'get_card'):
                    return card_db.get_card(cid)
                if isinstance(card_db, dict):
                     # Handle string/int keys
                     c = card_db.get(cid)
                     if c is None: c = card_db.get(str(cid))
                     return c
                return CardDatabase.get_card(cid)

            # 1. Pending Effects (Must resolve)
            if state.pending_effects:
                act = Action()
                act.type = ActionType.RESOLVE_EFFECT
                # The real engine might enforce which effect, but for stub we simplify
                # We usually resolve the top one (LIFO)
                # Just return one action
                return [act]

            player_id = state.active_player_id
            player = state.players[player_id]

            # 2. PASS
            pass_act = Action()
            pass_act.type = ActionType.PASS
            actions.append(pass_act)

            # 3. MANA_CHARGE
            # Allow charging from hand (simplified: all cards valid)
            for i, card in enumerate(player.hand):
                act = Action()
                act.type = ActionType.MANA_CHARGE
                act.source_instance_id = card.instance_id
                act.card_id = card.card_id
                act.target_player = player_id
                actions.append(act)

            # 4. PLAY_CARD
            # Calculate available mana
            mana_untapped = [c for c in player.mana_zone if not c.is_tapped]
            mana_count = len(mana_untapped)

            # Calculate available civs (from all mana cards, tapped or not usually counts for unlocking civ)
            # In DM, you need at least one card of that civ in mana zone to play a card.
            available_civs = set()
            for c in player.mana_zone:
                cdef = get_card_def(c.card_id)
                if cdef:
                    civs = cdef.get('civilizations', [])
                    if isinstance(civs, list):
                        for civ in civs: available_civs.add(civ)

            for i, card in enumerate(player.hand):
                cdef = get_card_def(card.card_id)
                if not cdef: continue

                cost = cdef.get('cost', 0)
                civs = cdef.get('civilizations', [])

                # Check cost
                if mana_count >= cost:
                    # Check civ
                    has_civ = False
                    if not civs:
                        has_civ = True # Colorless
                    else:
                        for civ in civs:
                            if civ in available_civs:
                                has_civ = True
                                break

                    if has_civ:
                        act = Action()
                        act.type = ActionType.PLAY_CARD
                        act.source_instance_id = card.instance_id
                        act.card_id = card.card_id
                        act.target_player = player_id
                        actions.append(act)

            # 5. ATTACK
            for i, card in enumerate(player.battle_zone):
                # Must be untapped and not sick
                if not card.is_tapped and not card.sick:
                    # Attack Player (Opponent)
                    act = Action()
                    act.type = ActionType.ATTACK_PLAYER
                    act.source_instance_id = card.instance_id
                    act.target_player = 1 - player_id
                    actions.append(act)

            return actions

    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, source_id: int, player_id: int, ctx: Any = None) -> None:
            try:
                from dm_toolkit.debug.effect_tracer import get_tracer, TraceEventType
                get_tracer().log_event(TraceEventType.COMMAND_EXECUTION, "CommandSystem.execute_command", {"cmd": str(cmd)})
            except ImportError:
                pass

            # Handle dictionary inputs
            cmd_type = None
            if hasattr(cmd, 'type'):
                cmd_type = cmd.type
            elif isinstance(cmd, dict):
                cmd_type = cmd.get('type')
                # Try mapping string type to Enum
                if isinstance(cmd_type, str):
                    try:
                        cmd_type = getattr(CommandType, cmd_type)
                    except AttributeError:
                        pass

            # Helper to access cmd attributes safely
            def get_attr(name, default=None):
                if hasattr(cmd, name): return getattr(cmd, name)
                if isinstance(cmd, dict): return cmd.get(name, default)
                return default

            if cmd_type == CommandType.TAP:
                target_filter = get_attr('target_filter')
                if target_filter:
                    zones = getattr(target_filter, 'zones', []) if hasattr(target_filter, 'zones') else target_filter.get('zones', [])
                    for z in zones:
                        if z in ["BATTLE_ZONE", "MANA_ZONE"]:
                            p = state.players[player_id]
                            zone_list = p.battle_zone if z == "BATTLE_ZONE" else p.mana_zone
                            for card in zone_list:
                                card.is_tapped = True

            elif cmd_type == CommandType.UNTAP:
                target_filter = get_attr('target_filter')
                if target_filter:
                    zones = getattr(target_filter, 'zones', []) if hasattr(target_filter, 'zones') else target_filter.get('zones', [])
                    for z in zones:
                        if z in ["BATTLE_ZONE", "MANA_ZONE"]:
                            p = state.players[player_id]
                            zone_list = p.battle_zone if z == "BATTLE_ZONE" else p.mana_zone
                            for card in zone_list:
                                card.is_tapped = False

            elif cmd_type == CommandType.RETURN_TO_HAND:
                target_filter = get_attr('target_filter')
                if target_filter:
                    zones = getattr(target_filter, 'zones', []) if hasattr(target_filter, 'zones') else target_filter.get('zones', [])
                    if "BATTLE_ZONE" in zones:
                         p = state.players[player_id]
                         while p.battle_zone:
                             c = p.battle_zone.pop(0)
                             p.hand.append(c)

            elif cmd_type == CommandType.SEARCH_DECK:
                 p = state.players[player_id]
                 if p.deck:
                     stub = p.deck.pop()
                     cid = stub.card_id if isinstance(stub, CardStub) else stub
                     state.add_card_to_hand(player_id, cid)

            elif cmd_type == CommandType.DESTROY:
                p = state.players[player_id]
                target_zone = p.battle_zone
                target_filter = get_attr('target_filter')
                if target_filter:
                    zones = getattr(target_filter, 'zones', []) if hasattr(target_filter, 'zones') else target_filter.get('zones', [])
                    if "MANA_ZONE" in zones: target_zone = p.mana_zone

                while target_zone:
                    c = target_zone.pop(0)
                    p.graveyard.append(c)

            elif cmd_type == CommandType.MANA_CHARGE:
                p = state.players[player_id]
                # Check explicit instance first
                instance_id = get_attr('instance_id') or get_attr('source_instance_id') or source_id

                found = False
                if instance_id > 0:
                    for i, c in enumerate(p.hand):
                        if c.instance_id == instance_id:
                            card = p.hand.pop(i)
                            card.is_tapped = False
                            p.mana_zone.append(card)
                            found = True
                            break

                if not found:
                    target_filter = get_attr('target_filter')
                    if target_filter and "HAND" in (getattr(target_filter, 'zones', []) if hasattr(target_filter, 'zones') else target_filter.get('zones', [])):
                        if p.hand:
                            c = p.hand.pop(0)
                            p.mana_zone.append(c)
                    elif p.deck:
                        stub = p.deck.pop()
                        cid = stub.card_id if isinstance(stub, CardStub) else stub
                        p.mana_zone.append(CardStub(cid, state.get_next_instance_id()))

            elif cmd_type == CommandType.DISCARD:
                p = state.players[player_id]
                if p.hand:
                    c = p.hand.pop(0)
                    p.graveyard.append(c)

            elif cmd_type == CommandType.BREAK_SHIELD:
                target_pid = 1 - player_id
                p = state.players[target_pid]
                if p.shield_zone:
                    c = p.shield_zone.pop(0)
                    p.hand.append(c)

            elif cmd_type == CommandType.DRAW_CARD:
                state.draw_cards(player_id, get_attr('amount', 1))

            elif cmd_type == CommandType.MOVE_CARD or cmd_type == CommandType.REPLACE_CARD_MOVE or cmd_type == CommandType.PLAY_FROM_ZONE:
                # Handle explicit moves
                instance_id = get_attr('instance_id') or get_attr('target_instance') or source_id
                to_zone = str(get_attr('to_zone', '')).upper()

                # Find card
                card = None
                owner_idx = -1
                from_zone_list = None

                for pid, pl in enumerate(state.players):
                    for zname in ['hand', 'mana_zone', 'battle_zone', 'shield_zone', 'graveyard', 'deck']:
                        zlist = getattr(pl, zname)
                        for i, c in enumerate(zlist):
                            if c.instance_id == instance_id:
                                card = c
                                owner_idx = pid
                                from_zone_list = zlist
                                break
                        if card: break
                    if card: break

                if card and from_zone_list is not None:
                    # Remove from old
                    from_zone_list.remove(card)

                    # Add to new
                    dest_p = state.players[owner_idx] # Keep owner unless changed

                    if to_zone in ['MANA', 'MANA_ZONE']:
                        card.is_tapped = False
                        dest_p.mana_zone.append(card)
                    elif to_zone in ['HAND']:
                        dest_p.hand.append(card)
                    elif to_zone in ['BATTLE', 'BATTLE_ZONE']:
                        if cmd_type == CommandType.PLAY_FROM_ZONE:
                            card.is_tapped = False
                            card.sick = True
                        dest_p.battle_zone.append(card)
                    elif to_zone in ['GRAVEYARD']:
                        dest_p.graveyard.append(card)
                    elif to_zone in ['SHIELD', 'SHIELD_ZONE']:
                        dest_p.shield_zone.append(card)
                    elif to_zone in ['DECK']:
                        dest_p.deck.append(card)

    class TokenConverter:
        @staticmethod
        def encode_state(state: Any, player_id: int, length: int) -> List[int]:
            return [0] * length

    class GenericCardSystem:
        @staticmethod
        def execute_card_effect(state: Any, card_id: int, player_id: int) -> None:
            """Stub implementation of effect resolution for Python mode."""
            try:
                from dm_toolkit.debug.effect_tracer import get_tracer, TraceEventType
                get_tracer().log_event(TraceEventType.EFFECT_RESOLUTION, "GenericCardSystem.execute_card_effect", {"card_id": card_id})
            except ImportError:
                pass

            card_data = CardDatabase.get_card(card_id)
            if not card_data:
                return

            # Determine effects based on Type (Spell vs Creature)
            ctype = card_data.get('type', 'SPELL')
            effects = card_data.get('effects', [])

            # Simple heuristic: execute all commands in the first relevant effect
            # In real engine, this is event-driven (ON_PLAY, ON_CAST_SPELL)

            trigger_key = 'ON_CAST_SPELL' if ctype == 'SPELL' else 'ON_PLAY'

            for effect in effects:
                if effect.get('trigger') == trigger_key:
                    commands = effect.get('commands', [])
                    for cmd_dict in commands:
                        # Map dict to CommandType and execute
                        cmd_type_str = cmd_dict.get('type', '')

                        # Create a dummy command object for CommandSystem
                        class CmdObj:
                            pass
                        cmd_obj = CmdObj()

                        # Simple mapping
                        if cmd_type_str == 'DRAW_CARD':
                            cmd_obj.type = CommandType.DRAW_CARD
                            cmd_obj.amount = cmd_dict.get('amount', 1)
                        elif cmd_type_str == 'MANA_CHARGE':
                            cmd_obj.type = CommandType.MANA_CHARGE
                        elif cmd_type_str == 'DESTROY':
                            cmd_obj.type = CommandType.DESTROY
                        elif cmd_type_str == 'DISCARD':
                            cmd_obj.type = CommandType.DISCARD
                        elif cmd_type_str == 'SEARCH_DECK':
                            cmd_obj.type = CommandType.SEARCH_DECK
                        else:
                            cmd_obj.type = CommandType.NONE

                        # Execute
                        if cmd_obj.type != CommandType.NONE:
                            CommandSystem.execute_command(state, cmd_obj, -1, player_id)

        @staticmethod
        def resolve_action(state: Any, action: Any, source_id: int) -> Any:
            try:
                from dm_toolkit.debug.effect_tracer import get_tracer, TraceEventType
                get_tracer().log_event(TraceEventType.EFFECT_RESOLUTION, "GenericCardSystem.resolve_action", {"action": str(action)})
            except ImportError:
                pass

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
                    action_to_resolve = state.pending_effects.pop()
                    # In a full engine, 'action_to_resolve' tracks which effect to run.
                    # Here we extract card_id and blindly run the 'primary' effect of that card.
                    card_id = getattr(action_to_resolve, 'card_id', 0)
                    if card_id > 0:
                        GenericCardSystem.execute_card_effect(state, card_id, player)

            elif is_type(atype, 'DRAW_CARD'):
                val = int(getattr(action, 'value1', 1))
                state.draw_cards(player, val)

            elif atype == ActionType.MANA_CHARGE:
                 if hasattr(action, 'card_id'):
                    cid = getattr(action, 'card_id')
                    p = state.players[player]
                    # Find and remove
                    for i, c in enumerate(p.hand):
                        if c.card_id == cid:
                            card = p.hand.pop(i)
                            # In DM, charged mana enters untapped but cannot be used immediately (handled by game logic limits, not tap state)
                            card.is_tapped = False
                            p.mana_zone.append(card)
                            break

            elif atype == ActionType.ATTACK_PLAYER:
                # Tap the attacker
                p = state.players[player]
                source_id = getattr(action, 'source_instance_id', -1)
                for c in p.battle_zone:
                    if c.instance_id == source_id:
                        c.is_tapped = True
                        break

            elif is_type(atype, 'CAST_SPELL') or atype == ActionType.PLAY_CARD:
                # Simulate removing card from hand when played
                cid = getattr(action, 'card_id', 0)
                inst_id = getattr(action, 'source_instance_id', 0)

                p = state.players[player]
                card_obj = None

                # Find and remove one instance from hand
                for i, c in enumerate(p.hand):
                    if c.card_id == cid:
                        card_obj = p.hand.pop(i)
                        break

                if card_obj is None:
                    # Fallback if card wasn't in hand (e.g. created/token)
                    card_obj = CardStub(cid, inst_id)

                # Check Type to decide destination
                cdata = CardDatabase.get_card(cid)
                ctype = cdata.get('type', 'SPELL')

                if ctype == 'CREATURE':
                    card_obj.sick = True
                    card_obj.is_tapped = False
                    p.battle_zone.append(card_obj)
                else:
                    # Spell goes to graveyard
                    state.players[player].graveyard.append(card_obj)

                # Add to pending effects stub
                state.pending_effects.append(action)

    class EffectResolver:
        @staticmethod
        def resolve_action(state: Any, action: Any, card_db: Any = None) -> None:
            # Adapt signature for GenericCardSystem
            # GenericCardSystem.resolve_action expects (state, action, player_id)
            pid = getattr(state, 'active_player_id', 0)
            GenericCardSystem.resolve_action(state, action, pid)

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

    class ActionGenerator:
        @staticmethod
        def generate_legal_actions(state: Any, card_db: Any) -> list:
            actions = []

            # Helper to get card data
            def get_card_def(cid):
                if hasattr(card_db, 'get_card'):
                    return card_db.get_card(cid)
                if isinstance(card_db, dict):
                     # Handle string/int keys
                     c = card_db.get(cid)
                     if c is None: c = card_db.get(str(cid))
                     return c
                return CardDatabase.get_card(cid)

            # 1. Pending Effects
            if state.pending_effects:
                act = Action()
                act.type = ActionType.RESOLVE_EFFECT
                actions.append(act)
                return actions

            pid = state.active_player_id
            player = state.players[pid]
            phase = state.current_phase

            # 2. Mana Phase (Phase 2)
            if phase == 2:
                # Mana Charge
                for card in player.hand:
                    act = Action()
                    act.type = ActionType.MANA_CHARGE
                    act.card_id = card.card_id
                    act.source_instance_id = card.instance_id
                    actions.append(act)

                # PASS
                pass_act = Action()
                pass_act.type = ActionType.PASS
                actions.append(pass_act)

            # 3. Main Phase (Phase 3)
            elif phase == 3:
                # Calculate usable mana and civs
                usable_mana = 0
                available_civs = set()
                for m in player.mana_zone:
                    if not m.is_tapped:
                        usable_mana += 1
                    # Tap or untap, mana provides civ
                    cdef = get_card_def(m.card_id)
                    if cdef:
                        civs = cdef.get('civilizations', [])
                        if isinstance(civs, list):
                            for civ in civs: available_civs.add(civ)

                # Play Cards
                for card in player.hand:
                    cdata = get_card_def(card.card_id)
                    if not cdata: continue

                    cost = cdata.get('cost', 9999)
                    card_civs = cdata.get('civilizations', [])

                    # Check civ requirement
                    has_civ = False
                    if not card_civs: has_civ = True
                    else:
                        for c in card_civs:
                            if c in available_civs:
                                has_civ = True
                                break

                    if cost <= usable_mana and has_civ:
                        act = Action()
                        act.type = ActionType.PLAY_CARD
                        act.card_id = card.card_id
                        act.source_instance_id = card.instance_id
                        actions.append(act)

                # PASS
                pass_act = Action()
                pass_act.type = ActionType.PASS
                actions.append(pass_act)

            # 4. Attack Phase (Phase 4)
            elif phase == 4:
                opponent_pid = 1 - pid
                opponent = state.players[opponent_pid]

                for card in player.battle_zone:
                    if not card.is_tapped and not card.sick:
                        # Attack Player
                        act = Action()
                        act.type = ActionType.ATTACK_PLAYER
                        act.source_instance_id = card.instance_id
                        act.target_player = opponent_pid
                        actions.append(act)

                        # Attack Tapped Creatures
                        for op_card in opponent.battle_zone:
                            if op_card.is_tapped:
                                act2 = Action()
                                act2.type = ActionType.ATTACK_CREATURE
                                act2.source_instance_id = card.instance_id
                                act2.target_player = opponent_pid
                                act2.value1 = op_card.instance_id
                                actions.append(act2)

                # PASS
                pass_act = Action()
                pass_act.type = ActionType.PASS
                actions.append(pass_act)

            else:
                # Default PASS
                pass_act = Action()
                pass_act.type = ActionType.PASS
                actions.append(pass_act)

            return actions

    class GameInstance:
        def __init__(self, game_id: int = 0):
            self.state = GameState()
            self.player_instances = self.state.players # Alias for simpler access in python tests
        def initialize(self): pass
        def start_game(self):
            self.state.setup_test_duel()
        def execute_action(self, action: Any) -> None:
            GenericCardSystem.resolve_action(self.state, action, 0)

    # Debugging functions for python stub
    def get_execution_context(state: Any) -> dict:
        if hasattr(state, 'execution_context'):
             return state.execution_context.variables
        return {}

    def get_command_details(cmd: Any) -> str:
        # Provide details string for command
        if hasattr(cmd, 'type'):
             return f"Type: {cmd.type}"
        return str(cmd)

    def get_pending_effects_info(state: Any) -> List[Any]:
        # Return summary tuple + command object for details
        # (type_str, source_id, controller, command_object)
        info = []
        for cmd in state.pending_effects:
             t = getattr(cmd, 'type', 'UNKNOWN')
             sid = getattr(cmd, 'source_instance_id', -1)
             pid = getattr(cmd, 'target_player', 0)
             info.append((str(t), sid, pid, cmd))
        return info

    class PhaseManager:
        @staticmethod
        def start_game(state: Any, db: Any) -> None:
            state.current_phase = 2  # Start at Mana Phase
            state.active_player_id = 0

            # Standard Setup: Shuffle, Shields, Hand
            import random
            for p in state.players:
                if not p.deck: continue

                # Shuffle
                random.shuffle(p.deck)

                # 5 Shields
                for _ in range(5):
                    if p.deck:
                        p.shield_zone.append(p.deck.pop())

                # 5 Hand
                for _ in range(5):
                    if p.deck:
                        p.hand.append(p.deck.pop())

        @staticmethod
        def next_phase(state: Any, db: Any) -> None:
            # Cycle 2 -> 3 -> 4 -> 5 (End) -> 2 (Next Turn)
            if state.current_phase == 2:
                state.current_phase = 3
            elif state.current_phase == 3:
                state.current_phase = 4
            elif state.current_phase == 4:
                state.current_phase = 5
            elif state.current_phase == 5:
                state.current_phase = 2
                state.turn_number += 1
                state.active_player_id = 1 - state.active_player_id
                # Untap all
                for p in state.players:
                    for c in p.battle_zone: c.is_tapped = False
                    for c in p.mana_zone: c.is_tapped = False
            else:
                # Fallback recovery
                state.current_phase = 2

        @staticmethod
        def check_game_over(state: Any, result: Any = None) -> Any:
            if state.game_over:
                if result is not None:
                     try:
                         result.is_over = True
                         result.result = state.winner
                     except Exception:
                         pass
                     return True, result
                return True
            return False

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
