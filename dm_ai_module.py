"""Minimal Python fallback shim for dm_ai_module used by tests and tools.

This file provides lightweight Python implementations of the engine-facing
symbols so the test-suite and tooling can import `dm_ai_module` even when
the native extension is not available. Implementations are intentionally
simple and only aim to satisfy common test and script usage paths.
"""

from __future__ import annotations

import json
import os
from enum import IntEnum
from typing import Any, List, Optional
import copy
import math
import uuid

# Minimal early fallbacks so `from dm_ai_module import GameInstance, GameResult`
# doesn't fail during partial/guarded imports. Real/full definitions follow
# later in the module and will override these when import completes.
class GameResult(IntEnum):
    NONE = -1
    P1_WIN = 0
    P2_WIN = 1
    DRAW = 2


class GameInstance:
    def __init__(self, seed: int = 0, card_db: Any = None):
        self.state = None
        self.card_db = card_db

# Try to load native extension if present in build output (prefer native C++ implementation)
# unless explicitly disabled via DM_DISABLE_NATIVE environment variable.
_disable_native = os.environ.get('DM_DISABLE_NATIVE', '').lower() in ('1', 'true', 'yes')
if not _disable_native:
    try:
        import importlib.util, importlib.machinery, sys
        # On Windows, proactively preload the onnxruntime DLL that ships with the
        # installed Python package. This avoids accidentally binding to a system-wide
        # onnxruntime.dll (e.g. under System32) which can be an older version and
        # trigger ORT API version mismatches when importing the native extension.
        try:
            if os.name == 'nt':
                import ctypes
                from pathlib import Path

                try:
                    import onnxruntime as _ort  # type: ignore

                    _capi_dir = Path(getattr(_ort, '__file__', '')).resolve().parent / 'capi'
                    _ort_dll = _capi_dir / 'onnxruntime.dll'
                    if _ort_dll.exists():
                        ctypes.WinDLL(str(_ort_dll))
                except Exception:
                    pass
        except Exception:
            pass
        
        _root = os.path.dirname(__file__)
        native_override = os.environ.get('DM_AI_MODULE_NATIVE')
        _candidates = []

        if native_override:
            try:
                if os.path.isdir(native_override):
                    for name in os.listdir(native_override):
                        if name.startswith('dm_ai_module') and (name.endswith('.pyd') or name.endswith('.so')):
                            _candidates.append(os.path.join(native_override, name))
                elif os.path.exists(native_override):
                    _candidates.append(native_override)
            except Exception:
                pass

        _candidates += [
            os.path.join(_root, 'bin', 'Release', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'build-msvc', 'Release', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'build-msvc', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'bin', 'dm_ai_module.cpython-312-x86_64-linux-gnu.so'),
            os.path.join(_root, 'build', 'dm_ai_module.cpython-312-x86_64-linux-gnu.so'), # Added build dir for linux
        ]
        _loaded_native = False
        for _p in _candidates:
            try:
                if _p and os.path.exists(_p):
                    loader = importlib.machinery.ExtensionFileLoader('dm_ai_module', _p)
                    spec = importlib.util.spec_from_loader('dm_ai_module', loader)
                    mod = importlib.util.module_from_spec(spec)
                    loader.exec_module(mod)
                    for _k, _v in mod.__dict__.items():
                        if _k.startswith('__'):
                            continue
                        globals()[_k] = _v
                    IS_NATIVE = True
                    _loaded_native = True
                    break
            except Exception:
                continue
    except Exception:
        IS_NATIVE = False
else:
    IS_NATIVE = False


try:
    import torch
    import numpy as np
except ImportError:
    pass

try:
    IS_NATIVE
except NameError:
    IS_NATIVE = False

# Expose a Python CommandEncoder fallback from native_prototypes if available
try:
    from native_prototypes.index_to_command.command_encoder import CommandEncoder
except Exception:
    class CommandEncoder:
        TOTAL_COMMAND_SIZE = 276

        @staticmethod
        def command_to_index(cmd: Any) -> int:
            try:
                if isinstance(cmd, dict):
                    t = cmd.get('type')
                else:
                    t = getattr(cmd, 'type', None)

                if isinstance(t, bytes):
                    try:
                        t = t.decode('utf-8')
                    except Exception:
                        t = str(t)

                if isinstance(t, str):
                    tt = t.upper()
                else:
                    tt = None

                # Lazy load CommandType if not available yet (in shim mode)
                CT = globals().get('CommandType')
                pass_val = getattr(CT, 'PASS', 5) if CT else 5
                mana_val = getattr(CT, 'MANA_CHARGE', 2) if CT else 2

                if tt == 'PASS' or t == 'PASS' or t == pass_val:
                    return 0
                if tt == 'MANA_CHARGE' or t == 'MANA_CHARGE' or t == mana_val:
                    return 1

                slot_candidates = []
                if isinstance(cmd, dict):
                    for k in ('slot_index', 'instance_id', 'source_instance_id', 'id'):
                        if k in cmd:
                            slot_candidates.append(cmd.get(k))
                else:
                    for k in ('slot_index', 'instance_id', 'source_instance_id', 'id'):
                        slot_candidates.append(getattr(cmd, k, None))

                for sc in slot_candidates:
                    if sc is None:
                        continue
                    try:
                        si = int(sc)
                    except Exception:
                        continue
                    try:
                        return 2 + (si % (CommandEncoder.TOTAL_COMMAND_SIZE - 2))
                    except Exception:
                        continue

                try:
                    s = json.dumps(cmd, sort_keys=True)
                except Exception:
                    s = str(cmd)
                return abs(hash(s)) % CommandEncoder.TOTAL_COMMAND_SIZE
            except Exception:
                return 0
else:
    _native_CommandEncoder = CommandEncoder

    class CommandEncoder:
        TOTAL_COMMAND_SIZE = getattr(_native_CommandEncoder, 'TOTAL_COMMAND_SIZE', 276)
        MANA_CHARGE_BASE = 1
        MANA_CHARGE_SLOTS = 19
        PLAY_FROM_ZONE_BASE = 20
        PLAY_FROM_ZONE_SLOTS = 256

        @staticmethod
        def index_to_command(idx: int):
            return _native_CommandEncoder.index_to_command(int(idx))

        @staticmethod
        def command_to_index(cmd: Any) -> int:
            try:
                return _native_CommandEncoder.command_to_index(cmd)
            except Exception:
                pass
            # Fallback logic omitted for brevity in hybrid mode as native usually handles it
            return 0


if not IS_NATIVE:
    class CommandType(IntEnum):
        NONE = 0
        PLAY_FROM_ZONE = 1
        MANA_CHARGE = 2
        TRANSITION = 3
        ATTACK = 4
        PASS = 5
        USE_ABILITY = 6
        SELECT_TARGET = 7
        ATTACK_PLAYER = 8
        ATTACK_CREATURE = 9
        BLOCK = 10
        RESOLVE_BATTLE = 11
        RESOLVE_PLAY = 12
        RESOLVE_EFFECT = 13
        BREAK_SHIELD = 14
        SHIELD_TRIGGER = 15
        CHOICE = 16
        SELECT_NUMBER = 17
        SHUFFLE_DECK = 18
        LOOK_AND_ADD = 19
        MEKRAID = 20
        REVEAL_CARDS = 21
        CAST_SPELL = 22
        SUMMON_TOKEN = 23
        SHIELD_BURN = 24
        LOOK_TO_BUFFER = 25
        REVEAL_TO_BUFFER = 26
        SELECT_FROM_BUFFER = 27
        PLAY_FROM_BUFFER = 28
        MOVE_BUFFER_TO_ZONE = 29
        FRIEND_BURST = 30
        REGISTER_DELAYED_EFFECT = 31
        MOVE_CARD = 32
        ADD_MANA = 33
        SEND_TO_MANA = 34
        PLAYER_MANA_CHARGE = 35
        SEARCH_DECK_BOTTOM = 36
        ADD_SHIELD = 37
        SEND_TO_DECK_BOTTOM = 38
        SEARCH_DECK = 39
        MUTATE = 40
        FLOW = 41
        QUERY = 42
        DRAW_CARD = 43
        DISCARD = 44
        DESTROY = 45
        BOOST_MANA = 46
        TAP = 47
        UNTAP = 48
        POWER_MOD = 49
        ADD_KEYWORD = 50
        RETURN_TO_HAND = 51


    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, source_id: int = -1, player_id: int = 0, ctx: Any = None) -> None:
            try:
                if hasattr(cmd, 'execute') and callable(getattr(cmd, 'execute')):
                    try:
                        cmd.execute(state)
                    except TypeError:
                        cmd.execute(state, ctx)
                    if not hasattr(state, 'command_history'):
                        state.command_history = []
                    state.command_history.append(cmd)
                    return

                if isinstance(cmd, dict):
                    t = cmd.get('type')
                    if t == 'PASS' or t == CommandType.PASS or t == 5:
                        if 'PhaseManager' in globals():
                            globals()['PhaseManager'].next_phase(state)
                        if not hasattr(state, 'command_history'):
                            state.command_history = []
                        state.command_history.append(cmd)
                        return

                    if t in (CommandType.MANA_CHARGE, 'MANA_CHARGE'):
                        pid = getattr(state, 'active_player_id', player_id)
                        # Enforce once-per-turn mana charge rule
                        mana_flags = getattr(state, 'mana_charged_this_turn', None)
                        if mana_flags is not None and pid < len(mana_flags) and mana_flags[pid]:
                            return  # Already charged this turn
                        cid = cmd.get('card_id') or cmd.get('instance_id') or cmd.get('source_instance_id') or 0
                        try:
                            hand = getattr(state.players[pid], 'hand', [])
                            removed = False
                            for i, c in enumerate(list(hand)):
                                try:
                                    if getattr(c, 'instance_id', None) == cid or getattr(c, 'card_id', None) == cid:
                                        try:
                                            hand.pop(i)
                                        except Exception:
                                            pass
                                        try:
                                            state.players[pid].mana_zone.append(c)
                                        except Exception:
                                            state.players[pid].mana_zone.append(CardStub(cid))
                                        removed = True
                                        break
                                except Exception:
                                    continue
                            if not removed:
                                state.players[pid].mana_zone.append(CardStub(cid))
                        except Exception:
                            pass
                        # Set flag: this player has charged this turn
                        if mana_flags is not None and pid < len(mana_flags):
                            mana_flags[pid] = True
                        if not hasattr(state, 'command_history'):
                            state.command_history = []
                        state.command_history.append(cmd)
                    # ... (Truncated other simplified command logic for brevity) ...
            except Exception:
                pass


    class CommandDef:
        def __init__(self):
            self.type = None
            self.amount = 0
            self.str_param = ''
            self.optional = False
            self.up_to = False
            self.instance_id = None
            self.target_instance = None
            self.owner_id = None
            self.from_zone = ''
            self.to_zone = ''
            self.mutation_kind = ''
            self.input_value_key = ''
            self.output_value_key = ''


    class FilterDef:
        def __init__(self):
            self.zones = []
            self.min_cost = None
            self.max_cost = None
            self.predicates = None


    class CardRegistry:
        @staticmethod
        def get_all_cards() -> dict:
            try:
                if 'JsonLoader' in globals():
                    return JsonLoader.load_cards('data/cards.json')
            except Exception:
                pass
            return {}


    class CardType(IntEnum):
        CREATURE = 0
        SPELL = 1


    class Phase(IntEnum):
        MANA = 2
        MAIN = 3
        ATTACK = 4
        END = 5


    class PlayerMode(IntEnum):
        AI = 0
        HUMAN = 1


    class Player:
        def __init__(self, player_id: int = 0):
            self.player_id = player_id
            self.hand: List[CardStub] = []
            self.mana_zone: List[CardStub] = []
            self.battle_zone: List[CardStub] = []
            self.graveyard: List[CardStub] = []
            self.shield_zone: List[CardStub] = []
            self.deck: List[int] = []
            self.life: int = 20


    class GameState:
        def __init__(self, seed: int = 0):
            self.players: List[Player] = [Player(0), Player(1)]
            try:
                self.current_phase = Phase.MANA
            except NameError:
                try:
                    self.current_phase = int(Phase.MANA)
                except Exception:
                    self.current_phase = 2
            self.active_player_id = 0
            self.pending_effects: List[Any] = []
            self.turn_number = 1
            self.game_over = False
            self.winner = -1
            self.command_history: List[Any] = []
            self.player_modes = [PlayerMode.AI, PlayerMode.AI]
            class _ExecCtx:
                def __init__(self):
                    self.variables: dict = {}
            try:
                self.execution_context = _ExecCtx()
            except Exception:
                self.execution_context = type('EC', (), {'variables': {}})()
            self.waiting_for_user_input = False
            self.pending_query = None
            self.mana_charged_this_turn = [False, False]

        def calculate_hash(self) -> int:
            return 0
        def create_snapshot(self) -> Any:
            return None
        def restore_snapshot(self, snap: Any) -> None:
            pass
        def apply_move(self, cmd: Any) -> None:
            pass
        def make_move(self, cmd: Any) -> None:
            pass
        def unmake_move(self) -> None:
            pass
        def get_next_instance_id(self) -> int:
            return 0
        def setup_test_duel(self):
            pass
        def is_human_player(self, player_id: int) -> bool:
            return False
        def add_card_to_hand(self, player: int, card_id: int, instance_id: Optional[int] = None, count: int = 1):
            pass
        def add_card_to_mana(self, player: int, card_id: int, count: int = 1):
            pass
        def set_deck(self, player: int, deck_ids: List[int], card_db: Any = None):
            pass
        def get_zone(self, player_id: int, zone_type: int) -> List[Any]:
            return []
        def add_test_card_to_battle(self, player: int, card_id: int, instance_id: int, tapped: bool = False, sick: bool = False):
            pass
        def get_pending_effects_info(self):
            return []
        def create_observer_view(self, observer_id: int):
            return self
        def clone(self):
            return self


    class GameInstance:
        def __init__(self, seed: int = 0, card_db: Any = None):
            self.state = GameState()
            self.card_db = card_db

        def start_game(self):
            pass
        def initialize_card_stats(self, deck_size: int):
            pass
        def execute_command(self, cmd):
            pass
        def step(self) -> bool:
            return False
        def resolve_command(self, cmd: Any) -> None:
            pass


    class IntentGenerator:
        @staticmethod
        def generate_legal_commands(state: GameState, card_db: Any = None) -> List[Any]:
            return []


    class PhaseManager:
        @staticmethod
        def start_game(state: GameState, card_db: Any = None) -> None:
            pass
        @staticmethod
        def setup_scenario(state: GameState, config: Any, card_db: Any = None) -> None:
            pass
        @staticmethod
        def next_phase(state: GameState, card_db: Any = None) -> None:
            pass
        @staticmethod
        def fast_forward(state: GameState, card_db: Any = None) -> None:
            pass
        @staticmethod
        def check_game_over(state: GameState, result_out: Any = None) -> tuple[bool, int]:
            return False, 0


    class GameResult(IntEnum):
        NONE = -1
        P1_WIN = 0
        P2_WIN = 1
        DRAW = 2


    class GameCommand:
        def __init__(self):
            self.type = None


    class EffectResolver:
        @staticmethod
        def resolve_action(state: GameState, action: Any, card_db: Any = None) -> None:
            pass


    class TensorConverter:
        @staticmethod
        def convert_to_tensor(state: Any, player_id: int, card_db: Any, mask_opponent: bool = True) -> List[float]:
            return [0.0] * 856


__all__ = [
    'IS_NATIVE', 'GameInstance', 'GameState', 'CommandType',
    'IntentGenerator', 'PhaseManager', 'EffectResolver',
    'CardType', 'Phase', 'GameResult', 'GameCommand', 'CommandSystem', 'ExecuteActionCompat',
]
if 'CardStub' in globals():
    __all__.append('CardStub')
if 'CommandEncoder' in globals():
    __all__.append('CommandEncoder')


if 'CardStub' not in globals():
    class CardStub:
        _iid = 1000

        def __init__(self, card_id: int, instance_id: Optional[int] = None, cost: int = 1, card_type: str = 'CREATURE'):
            if instance_id is None:
                CardStub._iid += 1
                instance_id = CardStub._iid
            self.card_id = card_id
            self.instance_id = instance_id
            self.is_tapped = False
            self.sick = False
            self.cost = cost
            self.card_type = card_type  # 'CREATURE' or 'SPELL'


# Helper: shim imports for other classes (Zone, DevTools etc) can be left at module level
# if they are not core engine classes that native replaces.

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
            return 0
        @staticmethod
        def trigger_loop_detection(state: Any):
            pass

if 'ParallelRunner' not in globals() and not IS_NATIVE:
    class ParallelRunner:
        def __init__(self, card_db: Any, sims: int, batch_size: int):
            self.card_db = card_db
        def play_games(self, initial_states: List[Any], evaluator_func: Any, temperature: float, add_noise: bool, threads: int) -> List[Any]:
            return []
    def create_parallel_runner(card_db: Any, sims: int, batch_size: int) -> Any:
        return ParallelRunner(card_db, sims, batch_size)

if 'JsonLoader' not in globals() and not IS_NATIVE:
    class JsonLoader:
        @staticmethod
        def load_cards(filepath: str) -> dict[int, Any]:
            return {}

if 'ExecuteActionCompat' not in globals():
    def ExecuteActionCompat(target: Any, action: Any, player_id: int = 0, ctx: Any = None) -> bool:
        try:
            if hasattr(target, 'execute_command'):
                target.execute_command(action)
                return True
        except Exception:
            pass
        return False

# Ensure core symbols exist on the module object even if a native extension
# was loaded earlier and didn't export them (Shim injection into Native module object)
try:
    import sys as _sys
    _mod = _sys.modules.get('dm_ai_module')
    if _mod is not None and IS_NATIVE:
        # If we are here, IS_NATIVE is True, so we imported native.
        # But we might want to inject pure-python helpers like `Zone` if native didn't have them.
        for _name, _obj in list(globals().items()):
             if _name.startswith('_'): continue
             if not hasattr(_mod, _name):
                 setattr(_mod, _name, _obj)
except Exception:
    pass
