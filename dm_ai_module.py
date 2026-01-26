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
from typing import Any, List, Optional


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _candidate_native_paths(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "bin", "dm_ai_module*.pyd"),
        os.path.join(root, "bin", "dm_ai_module*.so"),
        os.path.join(root, "build", "**", "dm_ai_module*.pyd"),
        os.path.join(root, "build", "**", "dm_ai_module*.so"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    return paths


def _load_native_in_place(module_name: str, path: str) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader(module_name, path)
    spec = importlib.util.spec_from_file_location(module_name, path, loader=loader)
    if spec is None:
        raise ImportError(f"Could not create spec for native module at: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _try_load_native() -> Optional[ModuleType]:
    root = _repo_root()
    override = os.environ.get("DM_AI_MODULE_NATIVE")
    candidates = [override] if override else _candidate_native_paths(root)
    for p in candidates:
        if not p:
            continue
        if os.path.isfile(p):
            try:
                return _load_native_in_place(__name__, p)
            except Exception:
                continue
    return None


_native = _try_load_native()
if _native is not None:
    try:
        globals().update(_native.__dict__)
    except Exception:
        _native = None

IS_NATIVE = (_native is not None)

# Provide minimal fallbacks only when missing from globals()
if 'Civilization' not in globals():
    class Civilization(Enum):
        FIRE = 1
        WATER = 2
        NATURE = 3
        LIGHT = 4
        DARKNESS = 5

if 'CardType' not in globals():
    class CardType(Enum):
        CREATURE = 1
        SPELL = 2

if 'ActionType' not in globals():
    class ActionType(IntEnum):
        PLAY_CARD = 1
        ATTACK_PLAYER = 2
        ATTACK_CREATURE = 3
        BLOCK_CREATURE = 4
        PASS = 5
        TAP = 10
        UNTAP = 11
        MANA_CHARGE = 7
        RESOLVE_EFFECT = 8

if 'Phase' not in globals():
    class Phase(IntEnum):
        START = 0
        DRAW = 1
        MANA = 2
        MAIN = 3
        ATTACK = 4
        END = 5

if 'CommandType' not in globals():
    class CommandType(Enum):
        NONE = 0
        TAP = 9
        UNTAP = 10
        RETURN_TO_HAND = 13
        MANA_CHARGE = 8
        DRAW_CARD = 5
        DESTROY = 7
        SEARCH_DECK = 15
        PLAY_FROM_ZONE = 25
        REPLACE_CARD_MOVE = 20
        MOVE_CARD = 19
        SELECT_TARGET = 30

if 'CardDatabase' not in globals():
    class CardDatabase:
        _cards = {}
        _loaded = False

        @staticmethod
        def load(path: str = "data/cards.json") -> None:
            if CardDatabase._loaded:
                return
            try:
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
            except Exception:
                pass

        @staticmethod
        def get_card(card_id: int) -> dict:
            if not CardDatabase._loaded:
                CardDatabase.load()
            return CardDatabase._cards.get(card_id, {})

if 'CardStub' not in globals():
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

if 'PlayerStub' not in globals():
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
        def shields(self):
            return len(self.shield_zone)

if 'ExecutionContext' not in globals():
    class ExecutionContext:
        def __init__(self):
            self.variables = {}

        def set_variable(self, name: str, value: Any):
            self.variables[name] = value

if 'GameState' not in globals():
    class GameState:
        def __init__(self, *args: Any, **kwargs: Any):
            self.game_over = False
            self.turn_number = 1
            self.players = [PlayerStub(), PlayerStub()]
            self.active_player_id = 0
            # Default winner value expected by tests/scripts is -1
            self.winner = -1
            self.current_phase = 0
            self.pending_effects: list[Any] = []
            self.instance_counter = 0
            self.execution_context = ExecutionContext()
            self.waiting_for_user_input = False
            self.pending_query = None
            self.command_history: list[Any] = []
            self.effect_buffer: list[Any] = []
            # Attach a card database to allow card type lookup in shims
            try:
                CardDatabase.load()
                self.card_db = CardDatabase
            except Exception:
                self.card_db = None

        def setup_test_duel(self) -> None:
            """Compatibility shim used by the GUI/tests to prepare a fresh game state.

            Performs a minimal reset of core attributes so higher-level helpers
            (e.g. `set_deck`, `PhaseManager.start_game`) can be called safely.
            """
            try:
                self.game_over = False
                self.turn_number = 1
                self.players = [PlayerStub(), PlayerStub()]
                self.active_player_id = 0
                self.winner = -1
                self.current_phase = 0
                self.pending_effects = []
                self.instance_counter = 0
                self.execution_context = ExecutionContext()
                self.waiting_for_user_input = False
                self.pending_query = None
                self.command_history = []
                self.effect_buffer = []
            except Exception:
                # Swallow errors to keep shim tolerant across environments
                pass

        def get_next_instance_id(self):
            self.instance_counter += 1
            return self.instance_counter

        def get_zone(self, player_id: int, zone_id: int) -> list:
            self._ensure_player(player_id)
            p = self.players[player_id]
            if zone_id == 0: return p.deck
            if zone_id == 1: return p.hand
            if zone_id == 2: return p.mana_zone
            if zone_id == 3: return p.battle_zone
            if zone_id == 4: return p.graveyard
            if zone_id == 5: return p.shield_zone
            return []

        def draw_cards(self, player_id: int, amount: int = 1) -> None:
            self._ensure_player(player_id)
            for _ in range(int(amount)):
                if not self.players[player_id].deck:
                    break
                stub = self.players[player_id].deck.pop()
                cid = stub.card_id if isinstance(stub, CardStub) else stub
                self.players[player_id].hand.append(CardStub(cid, self.get_next_instance_id()))

        def _ensure_player(self, player_id: int):
            while len(self.players) <= player_id:
                self.players.append(PlayerStub())

        def set_deck(self, player_id: int, deck_ids: List[int]) -> None:
            self._ensure_player(player_id)
            self.players[player_id].deck = [CardStub(cid, self.get_next_instance_id()) for cid in deck_ids]

        def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            inst = instance_id if instance_id != -1 else self.get_next_instance_id()
            self.players[player_id].deck.append(CardStub(card_id, inst))

        def add_card_to_hand(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            inst = instance_id if instance_id != -1 else self.get_next_instance_id()
            cs = CardStub(card_id, inst)
            for existing in self.players[player_id].hand:
                if getattr(existing, 'instance_id', None) == inst:
                    return
            self.players[player_id].hand.append(cs)

        def add_card_to_mana(self, player_id: int, card_id: int, instance_id: int = -1) -> None:
            self._ensure_player(player_id)
            inst = instance_id if instance_id != -1 else self.get_next_instance_id()
            self.players[player_id].mana_zone.append(CardStub(card_id, inst))

        def add_test_card_to_battle(self, player_id: int, card_id: int, instance_id: int, tapped: bool = False, sick: bool = False) -> Any:
            self._ensure_player(player_id)
            cs = CardStub(card_id, instance_id)
            cs.is_tapped = tapped
            cs.sick = sick
            for existing in self.players[player_id].battle_zone:
                if getattr(existing, 'instance_id', None) == instance_id:
                    existing.is_tapped = tapped
                    existing.sick = sick
                    return existing
            self.players[player_id].battle_zone.append(cs)
            return cs

        def get_card_instance(self, instance_id: int) -> Optional[CardStub]:
            target = int(instance_id)
            for p in self.players:
                for zone in [p.battle_zone, p.mana_zone, p.hand, p.shield_zone, p.graveyard, p.deck]:
                    for card in zone:
                        if isinstance(card, int):
                            continue
                        if getattr(card, 'instance_id', -1) == target or getattr(card, 'id', -1) == target:
                            return card
            return None

if 'Action' not in globals():
    class Action:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = None
            self.target_player = 0
            self.source_instance_id = 0
            self.card_id = 0
            self.slot_index = 0
            self.value1 = 0

if 'CommandDef' not in globals():
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

if 'GameInstance' not in globals():
    class GameInstance:
        def __init__(self, game_id: int = 0, card_db: Any = None):
            self.state = GameState()
            self.player_instances = self.state.players
            if card_db:
                self.state.card_db = card_db
        def initialize(self):
            pass
        def initialize_card_stats(self, deck_size: int):
            pass
        def get_card_stats(self):
            return {}
        def start_game(self):
            pass
        def execute_action(self, action: Any) -> None:
            # Forward to CommandSystem
            try:
                pid = getattr(action, 'target_player', getattr(self.state, 'active_player_id', 0))
                sid = getattr(action, 'source_instance_id', getattr(action, 'instance_id', 0))
                CommandSystem.execute_command(self.state, action, sid, pid)
            except Exception:
                pass

if 'CommandSystem' not in globals():
    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, source_id: int, player_id: int, ctx: Any = None) -> None:
            pass


if 'JsonLoader' not in globals():
    class JsonLoader:
        @staticmethod
        def load_cards(path: str) -> Any:
            try:
                import json
                if not os.path.exists(path):
                    root = _repo_root()
                    path = os.path.join(root, path)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Convert list to dict by id if needed
                if isinstance(data, list):
                    out = {}
                    for c in data:
                        try:
                            out[c.get('id')] = c
                        except Exception:
                            pass
                    return out
                return data
            except Exception:
                return {}

if 'ActionGenerator' not in globals():
    class ActionGenerator:
        @staticmethod
        def generate_legal_actions(state: Any, card_db: Any) -> list:
            return []

if 'PhaseManager' not in globals():
    class PhaseManager:
        @staticmethod
        def start_game(state: Any, db: Any) -> None:
            state.current_phase = 2
            state.active_player_id = 0
            import random
            for p in state.players:
                try:
                    random.shuffle(p.deck)
                except Exception:
                    pass
                for _ in range(5):
                    if getattr(p, 'deck', None):
                        try:
                            p.shield_zone.append(p.deck.pop())
                        except Exception:
                            pass
                for _ in range(5):
                    if getattr(p, 'deck', None):
                        try:
                            p.hand.append(p.deck.pop())
                        except Exception:
                            pass

        @staticmethod
        def next_phase(state: Any, db: Any) -> None:
            if getattr(state, 'current_phase', None) == 2:
                state.current_phase = 3
            elif state.current_phase == 3:
                state.current_phase = 4
            elif state.current_phase == 4:
                state.current_phase = 5
            elif state.current_phase == 5:
                state.current_phase = 2
                state.turn_number = getattr(state, 'turn_number', 0) + 1
                state.active_player_id = 1 - getattr(state, 'active_player_id', 0)
                for p in state.players:
                    for c in getattr(p, 'battle_zone', []):
                        try:
                            c.is_tapped = False
                        except Exception:
                            pass

        @staticmethod
        def check_game_over(state: Any, result: Any = None) -> Any:
            if getattr(state, 'game_over', False):
                if result is not None:
                    try:
                        result.is_over = True
                        result.result = getattr(state, 'winner', None)
                    except Exception:
                        pass
                    return True, result
                return True
            return False

if 'Zone' not in globals():
    from enum import IntEnum

    class Zone(IntEnum):
        DECK = 0
        HAND = 1
        MANA = 2
        BATTLE = 3
        GRAVEYARD = 4
        SHIELD = 5

if 'DevTools' not in globals():
    class DevTools:
        @staticmethod
        def move_cards(*args, **kwargs):
            """Flexible shim for DevTools.move_cards used by GUI/tests.

            Supported signatures (best-effort):
            - move_cards(gs, player_id, source, target, count=1, card_id_filter=-1)
            - move_cards(gs, instance_id, source, target)  # move specific instance
            """
            try:
                if len(args) < 4:
                    return 0
                gs = args[0]
                key = args[1]

                # Detect instance-id style call: args[1] is an instance id present in any zone
                def _find_card_by_instance(iid):
                    for pid, p in enumerate(getattr(gs, 'players', [])):
                        for zname in ('hand', 'deck', 'battle_zone', 'mana_zone', 'shield_zone', 'graveyard'):
                            z = getattr(p, zname, [])
                            for i, c in enumerate(list(z)):
                                try:
                                    if getattr(c, 'instance_id', None) == int(iid):
                                        return pid, zname, i, c
                                except Exception:
                                    continue
                    return None

                # If key matches an instance id, perform instance-based move
                try:
                    inst_lookup = _find_card_by_instance(int(key))
                except Exception:
                    inst_lookup = None

                # If found and call pattern matches (args[2] is source zone), move that instance
                if inst_lookup and (len(args) >= 4):
                    pid, from_zone_name, idx, card_obj = inst_lookup
                    target = args[3]
                    # map Zone to attr name
                    zone_map = {
                        Zone.DECK: 'deck',
                        Zone.HAND: 'hand',
                        Zone.MANA: 'mana_zone',
                        Zone.BATTLE: 'battle_zone',
                        Zone.GRAVEYARD: 'graveyard',
                        Zone.SHIELD: 'shield_zone',
                    }
                    dst_attr = zone_map.get(target, None)
                    if dst_attr is None:
                        return 0
                    try:
                        # remove from original zone
                        getattr(gs.players[pid], from_zone_name).pop(idx)
                    except Exception:
                        pass
                    try:
                        getattr(gs.players[pid], dst_attr).append(card_obj)
                    except Exception:
                        pass
                    return 1

                # Otherwise assume player-id style: gs, player_id, source, target, [count], [card_filter]
                try:
                    player_id = int(key)
                except Exception:
                    return 0

                src = args[2]
                dst = args[3]
                count = int(args[4]) if len(args) >= 5 else 1
                card_filter = int(args[5]) if len(args) >= 6 else -1

                zone_map = {
                    Zone.DECK: 'deck',
                    Zone.HAND: 'hand',
                    Zone.MANA: 'mana_zone',
                    Zone.BATTLE: 'battle_zone',
                    Zone.GRAVEYARD: 'graveyard',
                    Zone.SHIELD: 'shield_zone',
                }

                src_attr = zone_map.get(src, None)
                dst_attr = zone_map.get(dst, None)
                if src_attr is None or dst_attr is None:
                    return 0

                moved = 0
                p = gs.players[player_id]
                # iterate backwards to remove safely
                src_list = getattr(p, src_attr, [])
                for i in range(len(list(src_list)) - 1, -1, -1):
                    if moved >= count:
                        break
                    try:
                        card = src_list[i]
                        cid = getattr(card, 'card_id', None) or getattr(card, 'id', None) or card
                        if card_filter != -1 and int(cid) != int(card_filter):
                            continue
                        obj = src_list.pop(i)
                        try:
                            getattr(p, dst_attr).append(obj)
                        except Exception:
                            pass
                        moved += 1
                    except Exception:
                        continue
                return moved
            except Exception:
                return 0

        @staticmethod
        def trigger_loop_detection(state: Any):
            try:
                # Best-effort: create hash history attributes used by native code
                if not hasattr(state, 'hash_history'):
                    state.hash_history = []
                if not hasattr(state, 'calculate_hash'):
                    def _ch():
                        return 0
                    state.calculate_hash = _ch
                state.hash_history.append(getattr(state, 'calculate_hash')())
                state.hash_history.append(getattr(state, 'calculate_hash')())
                # optional update trigger
                try:
                    if hasattr(state, 'update_loop_check'):
                        state.update_loop_check()
                except Exception:
                    pass
            except Exception:
                pass

if 'ParallelRunner' not in globals():
    class ParallelRunner:
        def __init__(self, card_db: Any, sims: int, batch_size: int):
            self.card_db = card_db
            self.sims = sims
            self.batch_size = batch_size

        def play_games(self, initial_states: List[Any], evaluator_func: Any, temperature: float, add_noise: bool, threads: int) -> List[Any]:
            results = []
            for _ in initial_states:
                class Result:
                    def __init__(self):
                        self.result = 2
                        self.winner = 2
                        self.is_over = True
                results.append(Result())
            return results

        def play_deck_matchup(self, deck_a: List[int], deck_b: List[int], games: int, threads: int) -> List[int]:
            return [1] * games

    def create_parallel_runner(card_db: Any, sims: int, batch_size: int) -> Any:
        return ParallelRunner(card_db, sims, batch_size)

if 'GameResult' not in globals():
    class GameResult(IntEnum):
        NONE = -1
        P1_WIN = 0
        P2_WIN = 1
        DRAW = 2

if 'GameCommand' not in globals():
    class GameCommand:
        def __init__(self, *args: Any, **kwargs: Any):
            self.type = CommandType.NONE
            self.source_instance_id = -1
            self.target_player = -1
            self.card_id = -1

        def execute(self, state: Any) -> None:
            # Minimal no-op for tests that construct GameCommand
            return None

if 'FlowType' not in globals():
    class FlowType(IntEnum):
        NONE = 0
        SET_ATTACK_SOURCE = 1
        SET_ATTACK_PLAYER = 2
        SET_ATTACK_TARGET = 3
        PHASE_CHANGE = 4

if 'FlowCommand' not in globals():
    class FlowCommand:
        def __init__(self, flow_type: Any, new_value: Any, **kwargs: Any):
            self.flow_type = flow_type
            # legacy code sometimes expects `.type` or `.flow_type`
            try:
                self.type = flow_type
            except Exception:
                self.type = None
            self.new_value = new_value
            for k, v in kwargs.items():
                try:
                    setattr(self, k, v)
                except Exception:
                    pass

if 'MutationType' not in globals():
    class MutationType(IntEnum):
        TAP = 0
        UNTAP = 1
        POWER_MOD = 2
        ADD_KEYWORD = 3
        REMOVE_KEYWORD = 4

if 'MutateCommand' not in globals():
    class MutateCommand:
        def __init__(self, instance_id: int, mutation_type: Any, amount: int = 0, **kwargs: Any):
            self.instance_id = instance_id
            self.mutation_type = mutation_type
            self.amount = amount
            for k, v in kwargs.items():
                try:
                    setattr(self, k, v)
                except Exception:
                    pass

        if 'ActionGenerator' not in globals():
            class ActionGenerator:
                def __init__(self, registry: Any = None):
                    self.registry = registry

                def generate(self, state: Any, player_id: int) -> list:
                    # Minimal: return empty list of actions
                    return []

        if 'ActionEncoder' not in globals():
            class ActionEncoder:
                def __init__(self):
                    pass

                def encode(self, action: Any) -> dict:
                    # Minimal encoder returning a dict of basic fields
                    try:
                        return {
                            'type': getattr(action, 'type', None),
                            'card_id': getattr(action, 'card_id', None),
                            'source_instance_id': getattr(action, 'source_instance_id', getattr(action, 'instance_id', None))
                        }
                    except Exception:
                        return {}

        if 'EffectResolver' not in globals():
            class EffectResolver:
                @staticmethod
                def resolve(state: Any, effect: Any, player_id: int) -> None:
                    pass

        if 'TokenConverter' not in globals():
            class TokenConverter:
                def to_tokens(self, obj: Any) -> list:
                    # Return a short, deterministic token list for common objects.
                    try:
                        if obj is None:
                            return []
                        # already a token list
                        if isinstance(obj, list):
                            return [int(x) for x in obj][:256]
                        # card instance-like
                        if hasattr(obj, 'instance_id'):
                            return [int(getattr(obj, 'instance_id')) % 8192]
                        # dict-like
                        if isinstance(obj, dict):
                            tokens = []
                            for k, v in obj.items():
                                try:
                                    tokens.append(abs(hash(k)) % 8192)
                                    if isinstance(v, int):
                                        tokens.append(v % 8192)
                                    else:
                                        tokens.append(abs(hash(str(v))) % 8192)
                                except Exception:
                                    continue
                            return tokens[:256]
                        # fallback: hash of string
                        return [abs(hash(str(obj))) % 8192]
                    except Exception:
                        return []

                @staticmethod
                def get_vocab_size() -> int:
                    # Conservative default for Python shim; native may expose larger size.
                    return 8192

                @staticmethod
                def encode_state(state: Any, player_id: int, max_len: int = 512) -> list:
                    # Produce a compact token list representing player's zones and cards.
                    tokens: list[int] = []
                    try:
                        players = getattr(state, 'players', None)
                        if players is None or player_id >= len(players):
                            return tokens
                        p = players[player_id]
                        tokens.append(int(getattr(p, 'player_id', player_id)) % 8192)
                        for zone in ('hand', 'battle_zone', 'mana_zone', 'shield_zone', 'graveyard'):
                            z = getattr(p, zone, []) or []
                            tokens.append(len(z) % 8192)
                            for c in z:
                                cid = getattr(c, 'card_id', None) or getattr(c, 'base_id', None) or getattr(c, 'id', None)
                                if cid is None:
                                    tokens.append(abs(hash(str(c))) % 8192)
                                else:
                                    try:
                                        tokens.append(int(cid) % 8192)
                                    except Exception:
                                        tokens.append(abs(hash(str(cid))) % 8192)
                                if len(tokens) >= max_len:
                                    return tokens[:max_len]
                        return tokens[:max_len]
                    except Exception:
                        return tokens[:max_len]

        if 'TransitionCommand' not in globals():
            class TransitionCommand:
                def __init__(self, instance_id: int = -1, from_zone: str = '', to_zone: str = '', **kwargs: Any):
                    self.instance_id = instance_id
                    self.from_zone = from_zone
                    self.to_zone = to_zone
                    for k, v in kwargs.items():
                        try:
                            setattr(self, k, v)
                        except Exception:
                            pass

                def execute(self, state: Any) -> None:
                    # Minimal: attempt to move instance between zones by instance_id
                    try:
                        inst = state.get_card_instance(self.instance_id) if hasattr(state, 'get_card_instance') else None
                        if inst is None:
                            return
                        # naive removal from any zone and append to to_zone on owner 0
                        for p in state.players:
                            for zone_name in ('hand', 'battle_zone', 'mana_zone', 'shield_zone', 'graveyard', 'deck'):
                                z = getattr(p, zone_name, [])
                                for i, o in enumerate(list(z)):
                                    if getattr(o, 'instance_id', None) == getattr(inst, 'instance_id', None):
                                        try:
                                            z.pop(i)
                                        except Exception:
                                            pass
                        # append to active player's to_zone if possible
                        dest = 'graveyard' if 'GRAVE' in str(self.to_zone).upper() else ('battle_zone' if 'BATTLE' in str(self.to_zone).upper() else 'hand')
                        try:
                            state.players[getattr(state, 'active_player_id', 0)].__dict__.setdefault(dest, []).append(inst)
                        except Exception:
                            pass
                    except Exception:
                        pass
        def execute(self, state: Any) -> None:
            # Minimal mutate behavior used by tests: set tap/untap
            try:
                inst = state.get_card_instance(self.instance_id) if hasattr(state, 'get_card_instance') else None
                if inst is None:
                    return
                if getattr(self.mutation_type, 'name', None) == 'TAP' or str(self.mutation_type) == 'TAP':
                    inst.is_tapped = True
                if getattr(self.mutation_type, 'name', None) == 'UNTAP' or str(self.mutation_type) == 'UNTAP':
                    inst.is_tapped = False
            except Exception:
                pass

if 'DataCollector' not in globals():
    class DataCollector:
        def __init__(self, card_db: Any = None):
            self.card_db = card_db

        def collect_data_batch_heuristic(self, batch_size: int, include_history: bool, include_features: bool) -> Any:
            # Minimal stub returning an object with a `values` list attribute
            class Batch:
                def __init__(self):
                    self.values = []
            return Batch()

def get_card_stats(state: Any) -> Any:
    return {}

if 'DeckEvolutionConfig' not in globals():
    class DeckEvolutionConfig:
        def __init__(self):
            self.population_size = 10
            self.elites = 2
            self.mutation_rate = 0.1
            self.games_per_matchup = 2

if 'DeckEvolution' not in globals():
    class DeckEvolution:
        def __init__(self, card_db: Any):
            self.card_db = card_db
        def evolve_generation(self, population: List[Any], config: Any) -> List[Any]:
            return population

if 'HeuristicEvaluator' not in globals():
    class HeuristicEvaluator:
        def __init__(self, card_db: Any):
            self.card_db = card_db
        def evaluate(self, state: Any) -> Any:
            return [0.0]*600, 0.0

if 'ScenarioConfig' not in globals():
    class ScenarioConfig:
         def __init__(self):
             self.my_mana = 0
             self.my_hand_cards = []
             self.my_battle_zone = []
             self.my_mana_zone = []
             self.my_grave_yard = []
             self.my_shields = []
             self.enemy_shield_count = 5
             self.enemy_battle_zone = []
             self.enemy_can_use_trigger = False
