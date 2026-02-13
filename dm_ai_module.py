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

        # If the caller specifies an explicit native module path, try it first.
        # This lets launch scripts avoid manipulating PYTHONPATH (which can cause
        # Python to import the extension directly and skip this shim's DLL preloading).
        if native_override:
            try:
                if os.path.isdir(native_override):
                    # Accept a directory override and search within it.
                    for name in os.listdir(native_override):
                        if name.startswith('dm_ai_module') and name.endswith('.pyd'):
                            _candidates.append(os.path.join(native_override, name))
                elif os.path.exists(native_override):
                    _candidates.append(native_override)
            except Exception:
                pass

        _candidates += [
            os.path.join(_root, 'bin', 'Release', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'build-msvc', 'Release', 'dm_ai_module.cp312-win_amd64.pyd'),
            os.path.join(_root, 'build-msvc', 'dm_ai_module.cp312-win_amd64.pyd'),
        ]
        _loaded_native = False
        for _p in _candidates:
            try:
                if _p and os.path.exists(_p):
                    # Load with the canonical module name so the pyd's PyInit symbol matches
                    loader = importlib.machinery.ExtensionFileLoader('dm_ai_module', _p)
                    spec = importlib.util.spec_from_loader('dm_ai_module', loader)
                    mod = importlib.util.module_from_spec(spec)
                    loader.exec_module(mod)
                    # Copy public attributes into this module's globals
                    for _k, _v in mod.__dict__.items():
                        if _k.startswith('__'):
                            continue
                        globals()[_k] = _v
                    # Do NOT replace sys.modules['dm_ai_module'] with native mod.
                    # This Python file remains the canonical module; we just imported
                    # native symbols into its globals(). This allows Python fallbacks
                    # defined later in this file to augment the native extension.
                    IS_NATIVE = True
                    _loaded_native = True
                    break
            except Exception:
                continue
        # Continue execution even if native loaded; Python fallbacks below
        # will augment the native extension with additional helpers.
    except Exception:
        IS_NATIVE = False
else:
    # DM_DISABLE_NATIVE is set; skip native module loading
    IS_NATIVE = False


try:
    import torch
    import numpy as np
except ImportError:
    pass

# If native extension loader above didn't set `IS_NATIVE`, default to False
try:
    IS_NATIVE
except NameError:
    IS_NATIVE = False

# Expose a Python CommandEncoder fallback from native_prototypes if available
try:
    from native_prototypes.index_to_command.command_encoder import CommandEncoder
except Exception:
    # Provide a lightweight Python fallback for CommandEncoder so tooling
    # (head2head, training scripts) can map commands to canonical indices
    # even when the native extension is not available.
    class CommandEncoder:
        # Keep the canonical action space size used by exports and training
        TOTAL_COMMAND_SIZE = 276

        @staticmethod
        def command_to_index(cmd: Any) -> int:
            # Make mapping robust: never raise on malformed/None fields.
            # Accept either dict-like commands or objects with attributes
            try:
                if isinstance(cmd, dict):
                    t = cmd.get('type')
                else:
                    t = getattr(cmd, 'type', None)

                # Normalize simple string type values
                if isinstance(t, bytes):
                    try:
                        t = t.decode('utf-8')
                    except Exception:
                        t = str(t)

                if isinstance(t, str):
                    tt = t.upper()
                else:
                    tt = None

                # Common deterministic mappings for simple actions
                if tt == 'PASS' or t == 'PASS':
                    return 0
                if tt == 'MANA_CHARGE' or t == 'MANA_CHARGE':
                    return 1

                # PLAY_FROM_ZONE and similar: use slot_index / instance id when available
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
                    # Accept negative sentinel values but still map deterministically
                    try:
                        return 2 + (si % (CommandEncoder.TOTAL_COMMAND_SIZE - 2))
                    except Exception:
                        # fall back to next candidate
                        continue

                # Fallback: stable JSON/string hash into the canonical range
                try:
                    s = json.dumps(cmd, sort_keys=True)
                except Exception:
                    s = str(cmd)
                return abs(hash(s)) % CommandEncoder.TOTAL_COMMAND_SIZE
            except Exception:
                # As a last resort, return PASS index (safe default)
                return 0
else:
    # If we successfully imported a native CommandEncoder, wrap it to make
    # command_to_index more defensive: many code paths pass commands that
    # lack a `slot_index` key and the native implementation calls
    # `int(cmd.get('slot_index'))` which raises on None. Provide a thin
    # wrapper that tries the native implementation first and falls back to
    # safe heuristics (inspect instance_id, source_instance_id, id,
    # or stable hash) before returning PASS (0).
    _native_CommandEncoder = CommandEncoder

    class CommandEncoder:
        # Mirror prototype layout where possible
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
            # Try native implementation first
            try:
                return _native_CommandEncoder.command_to_index(cmd)
            except Exception:
                pass

            # Normalize type string
            if isinstance(cmd, dict):
                t = cmd.get('type')
            else:
                t = getattr(cmd, 'type', None)
            if isinstance(t, bytes):
                try:
                    t = t.decode('utf-8')
                except Exception:
                    t = str(t)
            tt = t.upper() if isinstance(t, str) else None

            # Explicit PASS mapping
            if tt == 'PASS' or t == 'PASS':
                return 0

            # Collect numeric candidates for slot/instance ids
            slot_candidates = []
            if isinstance(cmd, dict):
                for k in ('slot_index', 'instance_id', 'source_instance_id', 'id'):
                    if k in cmd:
                        slot_candidates.append(cmd.get(k))
            else:
                for k in ('slot_index', 'instance_id', 'source_instance_id', 'id'):
                    slot_candidates.append(getattr(cmd, k, None))

            # Handle MANA_CHARGE explicitly (prototype expects indices 1..19)
            if tt == 'MANA_CHARGE' or t == 'MANA_CHARGE':
                for sc in slot_candidates:
                    if sc is None:
                        continue
                    try:
                        si = int(sc)
                    except Exception:
                        continue
                    # clamp/validate into expected range
                    if si < CommandEncoder.MANA_CHARGE_BASE or si >= CommandEncoder.PLAY_FROM_ZONE_BASE:
                        si = CommandEncoder.MANA_CHARGE_BASE + (abs(si) % CommandEncoder.MANA_CHARGE_SLOTS)
                    return si
                # no numeric candidate: pick deterministic mana slot
                try:
                    s = json.dumps(cmd, sort_keys=True)
                except Exception:
                    s = str(cmd)
                return CommandEncoder.MANA_CHARGE_BASE + (abs(hash(s)) % CommandEncoder.MANA_CHARGE_SLOTS)

            # Handle PLAY_FROM_ZONE / PLAY
            if tt == 'PLAY_FROM_ZONE' or tt == 'PLAY' or t in ('PLAY_FROM_ZONE', 'PLAY'):
                for sc in slot_candidates:
                    if sc is None:
                        continue
                    try:
                        si = int(sc)
                    except Exception:
                        continue
                    # map into play slot window
                    idx = CommandEncoder.PLAY_FROM_ZONE_BASE + (si % CommandEncoder.PLAY_FROM_ZONE_SLOTS)
                    return idx
                # no numeric candidate: hash into play window
                try:
                    s = json.dumps(cmd, sort_keys=True)
                except Exception:
                    s = str(cmd)
                return CommandEncoder.PLAY_FROM_ZONE_BASE + (abs(hash(s)) % CommandEncoder.PLAY_FROM_ZONE_SLOTS)

            # Handle ATTACK commands: map deterministically into play window as fallback
            if tt == 'ATTACK' or t == 'ATTACK':
                for sc in slot_candidates:
                    if sc is None:
                        continue
                    try:
                        si = int(sc)
                    except Exception:
                        continue
                    idx = CommandEncoder.PLAY_FROM_ZONE_BASE + (si % CommandEncoder.PLAY_FROM_ZONE_SLOTS)
                    return idx
                try:
                    s = json.dumps(cmd, sort_keys=True)
                except Exception:
                    s = str(cmd)
                return CommandEncoder.PLAY_FROM_ZONE_BASE + (abs(hash(s)) % CommandEncoder.PLAY_FROM_ZONE_SLOTS)

            # Try to faux-call native encoder with a slot_index candidate
            for sc in slot_candidates:
                if sc is None:
                    continue
                try:
                    si = int(sc)
                except Exception:
                    continue
                try:
                    faux = {'type': t if isinstance(cmd, dict) else getattr(cmd, 'type', None), 'slot_index': si}
                    return _native_CommandEncoder.command_to_index(faux)
                except Exception:
                    try:
                        return 2 + (si % (CommandEncoder.TOTAL_COMMAND_SIZE - 2))
                    except Exception:
                        continue

            # Stable JSON/string hash fallback into full canonical range
            try:
                s = json.dumps(cmd, sort_keys=True)
            except Exception:
                s = str(cmd)
            try:
                return abs(hash(s)) % CommandEncoder.TOTAL_COMMAND_SIZE
            except Exception:
                return 0


class ActionType(IntEnum):
    NONE = 0
    PASS = 1
    MANA_CHARGE = 2
    PLAY_CARD = 3
    DECLARE_PLAY = 4
    PAY_COST = 5
    ATTACK_PLAYER = 6
    ATTACK_CREATURE = 7
    RESOLVE_EFFECT = 8


class PlayerIntent(IntEnum):
    """Player intent types (compatible with Action.type)."""
    NONE = 0
    PASS = 1
    MANA_CHARGE = 2
    PLAY_CARD = 3
    DECLARE_PLAY = 4
    PAY_COST = 5
    ATTACK_PLAYER = 6
    ATTACK_CREATURE = 7
    RESOLVE_EFFECT = 8


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
        # Minimal wrapper: attempt to call `execute` on cmd or treat as Action
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

            # Fallback: if cmd is dict-like, try to map basic commands
            if isinstance(cmd, dict):
                t = cmd.get('type')
                if t in (CommandType.MANA_CHARGE, 'MANA_CHARGE'):
                    # Emulate mana charge: remove from hand (if present) and add to mana_zone
                    pid = getattr(state, 'active_player_id', player_id)
                    # Prefer instance_id/source_instance_id/card_id
                    cid = cmd.get('card_id') or cmd.get('instance_id') or cmd.get('source_instance_id') or 0
                    try:
                        # Try to remove matching CardStub from hand by instance_id
                        hand = getattr(state.players[pid], 'hand', [])
                        removed = False
                        for i, c in enumerate(list(hand)):
                            try:
                                if getattr(c, 'instance_id', None) == cid or getattr(c, 'card_id', None) == cid:
                                    # remove and move to mana_zone
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
                            # No exact instance found: attempt to move any card with matching card_id
                            for i, c in enumerate(list(hand)):
                                try:
                                    if getattr(c, 'card_id', None) == cid:
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
                            # Fallback: append a new CardStub to mana_zone
                            state.players[pid].mana_zone.append(CardStub(cid))
                    except Exception:
                        try:
                            state.players[pid].mana_zone.append(CardStub(cid))
                        except Exception:
                            pass
                    if not hasattr(state, 'command_history'):
                        state.command_history = []
                    state.command_history.append(cmd)
                # DESTROY: remove cards by instance_id from execution_context.variables
                elif t == 'DESTROY' or t == getattr(CommandType, 'DESTROY', 'DESTROY'):
                    if not hasattr(state, 'execution_context'):
                        return
                    ctx_vars = getattr(state.execution_context, 'variables', {})
                    input_key = cmd.get('input_value_key') or 'selected'
                    target_ids = ctx_vars.get(input_key, [])
                    if not isinstance(target_ids, list):
                        target_ids = [target_ids]
                    
                    # Remove cards from all zones and move to graveyard
                    for pid in range(len(state.players)):
                        p = state.players[pid]
                        for zone_name in ('battle_zone', 'hand', 'mana_zone'):
                            zone = getattr(p, zone_name, [])
                            for tid in target_ids:
                                for c in list(zone):
                                    if getattr(c, 'instance_id', None) == tid:
                                        try:
                                            zone.remove(c)
                                            p.graveyard.append(c)
                                        except Exception:
                                            pass
                    if not hasattr(state, 'command_history'):
                        state.command_history = []
                    state.command_history.append(cmd)
                # SELECT_TARGET: place selected ids into execution_context.variables
                elif t == 'SELECT_TARGET' or t == getattr(CommandType, 'SELECT_TARGET', 'SELECT_TARGET'):
                    # Ensure execution_context exists and has variables mapping
                    if not hasattr(state, 'execution_context'):
                        class _EC:
                            def __init__(self):
                                self.variables = {}
                        state.execution_context = _EC()
                    ctx_vars = getattr(state.execution_context, 'variables', {})
                    # Determine candidates based on target_group
                    target_group = cmd.get('target_group', 'PLAYER_SELF')
                    amount = int(cmd.get('amount', 1) or 1)
                    out = []
                    pid = player_id
                    try:
                        if target_group == 'PLAYER_SELF':
                            zone = getattr(state.players[pid], 'battle_zone', []) or []
                            out = [getattr(c, 'instance_id', None) for c in zone][:amount]
                        elif target_group == 'PLAYER_OPPONENT':
                            zone = getattr(state.players[1-pid], 'battle_zone', []) or []
                            out = [getattr(c, 'instance_id', None) for c in zone][:amount]
                        else:
                            # Generic: use player's battle zone
                            zone = getattr(state.players[pid], 'battle_zone', []) or []
                            out = [getattr(c, 'instance_id', None) for c in zone][:amount]
                    except Exception:
                        out = []
                    key = cmd.get('output_value_key') or cmd.get('str_param') or 'selected'
                    try:
                        ctx_vars[key] = out
                    except Exception:
                        try:
                            setattr(state.execution_context, 'variables', {key: out})
                        except Exception:
                            pass
                    try:
                        state.execution_context.variables = ctx_vars
                    except Exception:
                        pass
                    if not hasattr(state, 'command_history'):
                        state.command_history = []
                    state.command_history.append(cmd)
        except Exception:
            pass


class CommandDef:
    def __init__(self):
        # Common command fields used by EngineCompat mapping
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
        # Try to use JsonLoader if present; otherwise return empty mapping
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


class Action:
    def __init__(self):
        self.type = ActionType.NONE
        self.card_id: Optional[int] = None
        self.source_instance_id: Optional[int] = None
        self.target_player: Optional[int] = None


class CardStub:
    _iid = 1000

    def __init__(self, card_id: int, instance_id: Optional[int] = None):
        if instance_id is None:
            CardStub._iid += 1
            instance_id = CardStub._iid
        self.card_id = card_id
        self.instance_id = instance_id
        self.is_tapped = False
        self.sick = False


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
        # Robustly initialize current_phase: if `Phase` isn't available
        # (e.g., native extension replaced module globals), fall back to
        # a local Phase-like mapping to avoid NameError during tests.
        try:
            self.current_phase = Phase.MANA
        except NameError:
            # Fallback to numeric phase value to avoid creating a new Enum type
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
        # Initialize player_modes: both AI by default
        self.player_modes = [PlayerMode.AI, PlayerMode.AI]
        # Lightweight execution context used by selection queries and UI
        class _ExecCtx:
            def __init__(self):
                self.variables: dict = {}
        try:
            self.execution_context = _ExecCtx()
        except Exception:
            self.execution_context = type('EC', (), {'variables': {}})()

    def calculate_hash(self) -> int:
        """Return a deterministic hash of key game-state fields for tests."""
        try:
            data = {
                'turn_number': int(getattr(self, 'turn_number', 0)),
                'active_player_id': int(getattr(self, 'active_player_id', 0)),
                'current_phase': int(getattr(self, 'current_phase', 0)) if not isinstance(getattr(self, 'current_phase', 0), str) else str(getattr(self, 'current_phase', 0)),
                'players': []
            }
            for p in getattr(self, 'players', []):
                pdata = {
                    'hand': [getattr(c, 'instance_id', None) for c in getattr(p, 'hand', [])],
                    'mana': [getattr(c, 'instance_id', None) for c in getattr(p, 'mana_zone', [])],
                    'battle': [getattr(c, 'instance_id', None) for c in getattr(p, 'battle_zone', [])],
                }
                data['players'].append(pdata)
            s = json.dumps(data, sort_keys=True)
            return abs(hash(s))
        except Exception:
            try:
                return abs(hash(repr(self)))
            except Exception:
                return 0

    def create_snapshot(self) -> Any:
        try:
            snap = type('Snap', (), {})()
            snap.state = copy.deepcopy(self)
            snap.hash_at_snapshot = self.calculate_hash()
            return snap
        except Exception:
            return None

    def restore_snapshot(self, snap: Any) -> None:
        try:
            if snap is None:
                return
            src = snap.state
            self.__dict__.clear()
            self.__dict__.update(copy.deepcopy(src.__dict__))
        except Exception:
            pass

    def apply_move(self, cmd: Any) -> None:
        """Execute a command (CommandDef or dict) using the command system."""
        try:
            # Unwrap CommandDef if wrapper
            if hasattr(cmd, '__dict__'):
                d = cmd.__dict__
            else:
                d = cmd

            # Delegate to CommandSystem
            if 'CommandSystem' in globals():
                CommandSystem.execute_command(self, d)
        except Exception:
            pass

    def make_move(self, action: Any) -> None:
        # Minimal move handling: record history and allow unmake via snapshot
        try:
            self._last_snap = self.create_snapshot()
            # Append to command history to indicate a change
            if not hasattr(self, 'command_history'):
                self.command_history = []
            self.command_history.append(action)
        except Exception:
            pass

    def unmake_move(self) -> None:
        try:
            if hasattr(self, '_last_snap') and self._last_snap is not None:
                self.restore_snapshot(self._last_snap)
                self._last_snap = None
        except Exception:
            pass

    def get_next_instance_id(self) -> int:
        """Return a deterministic increasing instance id for test usage."""
        if not hasattr(self, '_next_instance_id'):
            self._next_instance_id = 1000
        self._next_instance_id += 1
        return int(self._next_instance_id)

    def setup_test_duel(self):
        """Initialize game state for test duels.
        
        CRITICAL INITIALIZATION:
        Resets all game state to clean slate for testing.
        
        WHAT THIS DOES:
        1. Create fresh Player objects for both players
        2. Clear all zones (hand, mana, battle, shields, graveyard, deck)
        3. Reset game counters (turn number = 1)
        4. Set initial phase to MANA
        5. Set active player to 0 (first player)
        
        WHAT THIS DOES NOT DO:
        - Does NOT set up decks (use set_deck() afterwards)
        - Does NOT place initial shields/hand (use PhaseManager.start_game())
        - Does NOT load card database
        
        REGRESSION PREVENTION:
        - ALWAYS call this before set_deck() to ensure clean state
        - After setup_test_duel(), deck will be empty until you call set_deck()
        - After set_deck(), you must call PhaseManager.start_game() for shields/hand
        
        TYPICAL USAGE:
        ```python
        gs.setup_test_duel()                    # Clear everything
        gs.set_deck(0, [1,2,3,4,5]*8)          # Set P0 deck (40 cards)
        gs.set_deck(1, [1,2,3,4,5]*8)          # Set P1 deck (40 cards)
        PhaseManager.start_game(gs, card_db)    # Place shields + draw hand
        ```
        """
        # Ensure we have 2 players
        self.players = [Player(0), Player(1)]
        
        # Clear all zones
        for p in self.players:
            p.hand.clear()
            p.mana_zone.clear()
            p.battle_zone.clear()
            p.shield_zone.clear()
            p.graveyard.clear()
            p.deck.clear()
        
        # Reset game state
        self.turn_number = 1
        self.active_player_id = 0
        # Robustly set phase to MANA even if Phase symbol missing
        try:
            self.current_phase = Phase.MANA
        except NameError:
            try:
                self.current_phase = int(getattr(Phase, 'MANA', 2))
            except Exception:
                self.current_phase = 2
        self.game_over = False
        self.winner = -1

    def is_human_player(self, player_id: int) -> bool:
        """Check if a player is human."""
        if 0 <= player_id < len(self.player_modes):
            return self.player_modes[player_id] == PlayerMode.HUMAN
        return False

    def add_card_to_hand(self, player: int, card_id: int, instance_id: Optional[int] = None, count: int = 1):
        """
        Add card(s) to a player's hand.

        Backwards-compatible helper:
        - If `instance_id` is provided (positional 3rd arg in many tests), a single
          CardStub is created with that instance id and returned.
        - Otherwise, `count` cards are created (default 1) with generated instance ids.
        """
        if instance_id is not None and count == 1:
            c = CardStub(card_id, instance_id)
            self.players[player].hand.append(c)
            return c

        for _ in range(count):
            c = CardStub(card_id)
            self.players[player].hand.append(c)
        try:
            return self.players[player].hand[-1]
        except Exception:
            return None

    def add_card_to_mana(self, player: int, card_id: int, count: int = 1):
        for _ in range(count):
            c = CardStub(card_id)
            self.players[player].mana_zone.append(c)

    def set_deck(self, player: int, deck_ids: List[int]):
        """Set a player's deck from a list of card IDs.
        
        CRITICAL DECK INITIALIZATION:
        - Deck must contain CardStub objects, NOT raw card IDs (integers)
        - Each CardStub must have a unique instance_id for tracking across zones
        - Instance IDs should start from a high base (10000+) to avoid conflicts
        
        REGRESSION PREVENTION:
        - DO NOT assign deck_ids directly: self.players[player].deck = deck_ids
        - ALWAYS convert IDs to CardStub objects with unique instance_ids
        - Without CardStub objects, card movement between zones will fail
        
        Args:
            player: Player index (0 or 1)
            deck_ids: List of card IDs to create deck from
        """
        try:
            # Convert card IDs to CardStub objects with unique instance IDs
            # Start instance IDs from a high number to avoid conflicts with cards in other zones
            base_instance_id = 10000 + (player * 1000)
            deck_cards = []
            for i, card_id in enumerate(deck_ids):
                instance_id = base_instance_id + i
                deck_cards.append(CardStub(card_id, instance_id))
            self.players[player].deck = deck_cards
        except Exception:
            pass

    def get_zone(self, player_id: int, zone_type: int) -> List[Any]:
        try:
            p = self.players[player_id]
            zones = [p.deck, p.hand, p.mana_zone, p.battle_zone, p.graveyard, p.shield_zone]
            if 0 <= zone_type < len(zones):
                return zones[zone_type]
            return []
        except Exception:
            return []

    def add_test_card_to_battle(self, player: int, card_id: int, instance_id: int, tapped: bool = False, sick: bool = False):
        c = CardStub(card_id, instance_id)
        c.is_tapped = tapped
        c.sick = sick
        self.players[player].battle_zone.append(c)
        return c

    def get_pending_effects_info(self):
        return list(self.pending_effects)

    def create_observer_view(self, observer_id: int):
        view = self.clone()
        opponent_id = 1 - observer_id
        if 0 <= opponent_id < len(view.players):
            view.players[opponent_id].hand = [CardStub(0, c.instance_id) for c in view.players[opponent_id].hand]
        return view

    def clone(self):
        return copy.deepcopy(self)


class GameInstance:
    def __init__(self, seed: int = 0, card_db: Any = None):
        self.state = GameState()
        self.card_db = card_db

    def start_game(self):
        """Initialize game to starting state."""
        # Ensure Phase is available in global scope (may be overridden by native)
        _Phase = globals().get('Phase')
        if _Phase is not None:
            self.state.current_phase = _Phase.MANA
        else:
            # Fallback to integer value if Phase enum not available
            self.state.current_phase = 2  # Phase.MANA = 2
        self.state.active_player_id = 0

    def initialize_card_stats(self, deck_size: int):
        pass

    def execute_action(self, action):
        """Execute a game action (mana charge, play card, etc.).
        
        CRITICAL: Executes commands using CommandSystem with proper fallback chain.
        Delegates to CommandSystem → EngineCompat → legacy execute_action.
        
        Args:
            action: Can be either an Action object or a dict-format command
        """
        try:
            # Normalize action to dict format if needed
            if not isinstance(action, dict):
                # Convert Action object to dict-like command
                action_type = getattr(action, 'type', None)
                # Normalize ActionType enum to integer or string
                if hasattr(action_type, 'value'):
                    action_type = action_type.value
                elif hasattr(action_type, 'name'):
                    action_type = action_type.name
                
                action_dict = {
                    'type': action_type,
                    'card_id': getattr(action, 'card_id', None),
                    'source_instance_id': getattr(action, 'source_instance_id', None),
                    'instance_id': getattr(action, 'source_instance_id', None),
                    'target_player': getattr(action, 'target_player', None),
                    'player_id': getattr(action, 'target_player', getattr(self.state, 'active_player_id', 0))
                }
            else:
                action_dict = action
            
            # Try new Command API (apply_move) first if available (Native Binding)
            if hasattr(self.state, 'apply_move'):
                try:
                    self.state.apply_move(action_dict)
                    return
                except Exception:
                    pass

            # Try CommandSystem first
            _CommandSystem = globals().get('CommandSystem')
            if _CommandSystem is not None:
                try:
                    _CommandSystem.execute_command(
                        self.state, 
                        action_dict, 
                        source_id=action_dict.get('source_instance_id', -1),
                        player_id=action_dict.get('player_id', getattr(self.state, 'active_player_id', 0))
                    )
                    return
                except Exception:
                    pass
            
            # Fallback to EngineCompat if CommandSystem unavailable
            try:
                from dm_toolkit.engine.compat import EngineCompat
                EngineCompat.ExecuteCommand(self.state, action_dict, self.card_db)
                return
            except Exception:
                pass
        except Exception:
            pass

    def step(self) -> bool:
        """
        Execute one step of game progression.
        
        CRITICAL GAME LOOP IMPLEMENTATION:
        1. Generate legal actions for current game state
        2. Execute first action (AI behavior)
        3. Call fast_forward() to auto-advance phases without player input
        
        AI ACTION SELECTION:
        - Prefer meaningful actions (mana charge, play card, attack) over PASS
        - PASS should only be chosen when no other actions are available
        - This ensures the game progresses (mana is charged, creatures are played, etc.)
        
        REGRESSION PREVENTION - MANA CHARGE BUG:
        - DO NOT always execute actions[0] - it may be PASS
        - ALWAYS filter out PASS actions first, prefer non-PASS actions
        - Without this, AI will never charge mana or play cards
        - Test: After multiple turns, players should have mana in mana_zone
        
        REGRESSION PREVENTION - GAME PROGRESSION:
        - Always call PhaseManager.fast_forward() after executing an action
        - This ensures game progresses through automatic phases (draw, untap, etc.)
        - Without fast_forward(), game will be stuck waiting for input in every phase
        
        Returns:
            bool: True if successful, False if game is over or no actions available
        """
        # Check game over
        if getattr(self.state, 'game_over', False):
            return False
        
        # Generate legal actions for current state
        try:
            actions = ActionGenerator.generate_legal_actions(self.state, self.card_db)
        except Exception:
            actions = []
        
        if not actions:
            return False
        
        # CRITICAL: Prefer non-PASS actions over PASS
        # Filter actions to find meaningful actions (mana charge, play, attack, etc.)
        non_pass_actions = []
        for action in actions:
            action_type = action.get('type') if isinstance(action, dict) else getattr(action, 'type', None)
            # Skip PASS actions (ActionType.PASS = 1)
            try:
                from dm_toolkit.dm_types import ActionType
                if action_type != ActionType.PASS and action_type != 1:
                    non_pass_actions.append(action)
            except Exception:
                # Fallback: if we can't import ActionType, just check numeric value
                if action_type != 1:  # 1 is ActionType.PASS
                    non_pass_actions.append(action)
        
        # Choose action: prefer non-PASS, fall back to PASS if no other options
        action_to_execute = non_pass_actions[0] if non_pass_actions else actions[0]
        
        # Execute the chosen action
        try:
            self.execute_action(action_to_execute)
            
            # CRITICAL: Call fast_forward to auto-advance through phases
            # This moves the game from current phase to the next player-input-required state
            # Without this, the game will be stuck in the current phase
            PhaseManager.fast_forward(self.state, self.card_db)
            
            return True
        except Exception:
            return False

    def resolve_action(self, action: Any) -> None:
        """
        Resolve/execute a specific action.
        
        Args:
            action: The action to execute
        """
        try:
            self.execute_action(action)
        except Exception:
            pass


class ActionEncoder:
    @staticmethod
    def action_to_index(action: Any) -> int:
        try:
            # Prefer canonical CommandEncoder mapping when available
            if 'CommandEncoder' in globals() and CommandEncoder is not None:
                try:
                    # If action is already a dict matching CommandEncoder expectations
                    if isinstance(action, dict):
                        return CommandEncoder.command_to_index(action)
                    # If action is an Action-like object, attempt to convert
                    t = getattr(action, 'type', None)
                    if t is not None:
                        # Create a minimal dict representation and delegate
                        cmd = {}
                        # Map ActionType.PASS/MANA_CHARGE/PLAY_CARD to expected CommandEncoder types
                        # Use string types for compatibility with older CommandEncoder implementations
                        if t == ActionType.PASS:
                            cmd['type'] = 'PASS'
                        elif t == ActionType.MANA_CHARGE:
                            cmd['type'] = 'MANA_CHARGE'
                            # slot_index is not known from Action; best-effort fallback to 1
                            cmd['slot_index'] = getattr(action, 'source_instance_id', 1) or 1
                        elif t == ActionType.PLAY_CARD:
                            cmd['type'] = 'PLAY_FROM_ZONE'
                            cmd['slot_index'] = getattr(action, 'source_instance_id', 0) or 0
                        else:
                            # Unknown mapping, fall back to hashed index
                            raise ValueError('no command mapping for action type')
                        return CommandEncoder.command_to_index(cmd)
                except Exception:
                    # Fall through to legacy hash-based mapping
                    pass

            key = (getattr(action, 'type', 0), getattr(action, 'card_id', -1), getattr(action, 'source_instance_id', -1))
            return abs(hash(key)) % 1024
        except Exception:
            return -1


class ActionGenerator:
    @staticmethod
    def generate_legal_actions(state: GameState, card_db: Any = None) -> List[Action]:
        out: List[Action] = []
        try:
            pid = getattr(state, 'active_player_id', 0)
            p = state.players[pid]

            # PASS is always legal
            a = Action()
            a.type = ActionType.PASS
            out.append(a)

            phase = getattr(state, 'current_phase', Phase.MANA)

            if phase == Phase.MANA:
                for c in list(p.hand):
                    ma = Action()
                    ma.type = ActionType.MANA_CHARGE
                    ma.card_id = c.card_id
                    ma.source_instance_id = c.instance_id
                    out.append(ma)

            elif phase == Phase.MAIN:
                for c in list(p.hand):
                    pa = Action()
                    pa.type = ActionType.PLAY_CARD
                    pa.card_id = c.card_id
                    pa.source_instance_id = c.instance_id
                    out.append(pa)

            elif phase == Phase.ATTACK:
                 # Minimal attack logic for fallback
                for c in list(p.battle_zone):
                     if not c.is_tapped and not c.sick:
                        att = Action()
                        att.type = ActionType.ATTACK_PLAYER
                        att.source_instance_id = c.instance_id
                        att.target_player = 1 - pid
                        out.append(att)

        except Exception:
            return []
        return out

    @staticmethod
    def generate_legal_commands(state: GameState, card_db: Any = None) -> List[Any]:
        # Prefer native command-first binding if present
        try:
            gen = globals().get('generate_commands', None)
            if callable(gen):
                return gen(state, card_db) or []
        except Exception:
            pass

        # Fallback: map legacy Action objects to minimal command dicts
        actions = ActionGenerator.generate_legal_actions(state, card_db) or []
        out: List[Any] = []
        for a in actions:
            try:
                t = getattr(a, 'type', None)
                if t == ActionType.PASS:
                    typ = 'PASS'
                elif t == ActionType.MANA_CHARGE:
                    typ = 'MANA_CHARGE'
                elif t == ActionType.PLAY_CARD:
                    typ = 'PLAY_FROM_ZONE'
                elif t == ActionType.ATTACK_PLAYER:
                    typ = 'ATTACK'
                else:
                    typ = str(t)
                cmd = {'type': typ, 'uid': str(uuid.uuid4())}
                iid = getattr(a, 'instance_id', None) or getattr(a, 'source_instance_id', None)
                if iid is not None:
                    cmd['instance_id'] = iid
                cid = getattr(a, 'card_id', None)
                if cid is not None:
                    cmd['card_id'] = cid
                out.append(cmd)
            except Exception:
                continue
        return out


class IntentGenerator(ActionGenerator):
    pass


class PhaseManager:
    @staticmethod
    def start_game(state: GameState, card_db: Any = None) -> None:
        """Start the game with initial setup: 5 shields and 5 hand cards for each player.
        
        CRITICAL GAME INITIALIZATION:
        This method performs the standard Duel Masters game setup:
        1. Set initial phase to MANA and active player to 0
        2. Place 5 cards from top of each player's deck as shields (face-down)
        3. Draw 5 cards into each player's hand
        
        REGRESSION PREVENTION:
        - MUST place shields BEFORE drawing cards (correct order)
        - MUST move actual CardStub objects from deck, not create new ones
        - Use deck.pop() to take from top of deck (end of list)
        - After setup: deck should have 30 cards (40 - 5 shields - 5 hand)
        - DO NOT call add_card_to_hand() with placeholder IDs - move real cards
        
        Args:
            state: GameState to initialize
            card_db: Card database (optional, for future use)
        """
        try:
            state.current_phase = Phase.MANA
            state.active_player_id = 0
            
            # Initial setup: Place 5 shields and draw 5 cards for each player
            for pid in (0, 1):
                p = state.players[pid]
                
                # Place 5 shields from top of deck (face-down protection)
                shields_to_place = min(5, len(p.deck))
                for _ in range(shields_to_place):
                    if p.deck:
                        card = p.deck.pop()  # Take from top (end of list)
                        p.shield_zone.append(card)
                
                # Draw 5 cards into hand (starting hand)
                cards_to_draw = min(5, len(p.deck))
                for _ in range(cards_to_draw):
                    if p.deck:
                        card = p.deck.pop()  # Take from top (end of list)
                        p.hand.append(card)
        except Exception:
            pass

    @staticmethod
    def setup_scenario(state: GameState, config: Any, card_db: Any = None) -> None:
        pass

    @staticmethod
    def next_phase(state: GameState, card_db: Any = None) -> None:
        """Advance to the next game phase.
        
        PHASE PROGRESSION:
        MANA → MAIN → ATTACK → END → (next turn) MANA
        
        TURN TRANSITION (END → MANA):
        When transitioning from END to MANA, performs turn start procedures:
        1. Switch active player (0 ↔ 1)
        2. Untap all cards in mana zone and battle zone
        3. Remove summoning sickness (creatures can attack this turn)
        4. Draw 1 card from deck into hand
        5. Increment turn counter when P0 becomes active
        
        REGRESSION PREVENTION:
        - When drawing at turn start, MUST move actual CardStub from deck
        - DO NOT use add_card_to_hand(player_id, placeholder_id)
        - Use deck.pop() to get card, then hand.append(card)
        - This preserves instance_id and card_id for tracking
        
        Args:
            state: GameState to advance
            card_db: Card database (optional)
        """
        try:
            # Compare by numeric value to tolerate different Enum subclasses or ints
            def _phase_value(x):
                # Robust numeric extraction from many possible phase representations
                try:
                    return int(x)
                except Exception:
                    pass
                try:
                    return int(getattr(x, 'value'))
                except Exception:
                    pass
                try:
                    return int(str(x))
                except Exception:
                    return int(getattr(Phase, 'END', 5))

            cur = _phase_value(state.current_phase)
            
            # Get Phase enum from globals() safely
            _Phase = globals().get('Phase')
            if _Phase is None:
                # Fallback: use integer values directly
                class _Phase(IntEnum):
                    MANA = 2
                    MAIN = 3
                    ATTACK = 4
                    END = 5

            if cur == int(_Phase.MANA):
                state.current_phase = _Phase.MAIN
            elif cur == int(_Phase.MAIN):
                state.current_phase = _Phase.ATTACK
            elif cur == int(_Phase.ATTACK):
                state.current_phase = _Phase.END
            else:
                # END -> Next Turn (MANA)
                state.active_player_id = 1 - state.active_player_id
                state.current_phase = _Phase.MANA

                # Untap Step: Reset all tapped cards
                p = state.players[state.active_player_id]
                for c in p.mana_zone:
                    c.is_tapped = False
                for c in p.battle_zone:
                    c.is_tapped = False
                    c.sick = False  # Remove summoning sickness

                # Draw Step: Draw 1 card from deck (if available)
                # CRITICAL: Move actual CardStub from deck, not placeholder
                p = state.players[state.active_player_id]
                if p.deck:
                    card = p.deck.pop()  # Take from top of deck
                    p.hand.append(card)  # Add to hand

                # Increment Turn Counter
                # Standard practice: increment when player 0 becomes active
                if state.active_player_id == 0:
                    state.turn_number += 1
        except Exception:
            pass

    @staticmethod
    def fast_forward(state: GameState, card_db: Any = None) -> None:
        """Auto-advance game state through phases that don't require player input.
        
        CRITICAL AUTO-PROGRESSION:
        This method handles automatic phase transitions and turn progression.
        Called after each player action to advance to next input-required state.
        
        PHASE FLOW:
        - MANA → MAIN: Automatic (mana phase has no automation, just transitions)
        - MAIN → ATTACK: After player passes main phase
        - ATTACK → END: After attack declarations/blocks
        - END → Next turn MANA: Turn end, switch active player
        
        TURN TRANSITIONS (END → MANA):
        1. Switch active player (0 ↔ 1)
        2. Untap all mana and creatures
        3. Remove summoning sickness from creatures
        4. Draw 1 card from deck
        5. Increment turn counter (when P0 becomes active)
        
        REGRESSION PREVENTION:
        - ALWAYS advance at least one phase to prevent infinite loops
        - STOP at MAIN phase (requires player input for actions)
        - STOP if pending_effects exist (need resolution before proceeding)
        - When drawing at turn start, move actual CardStub from deck, not placeholder
        - Use deck.pop() to take from top, then hand.append() - preserve instance_id
        
        Args:
            state: GameState to advance
            card_db: Card database (optional, for future use)
        """
        try:
            # Ensure the current_phase is a valid Phase; normalize unknown values
            cp = getattr(state, 'current_phase', None)
            try:
                if not isinstance(cp, Phase):
                    state.current_phase = Phase(cp)
            except Exception:
                state.current_phase = Phase.MANA

            # If there are pending effects, don't auto-advance; let resolver handle them
            if getattr(state, 'pending_effects', None):
                return

            # Advance the phase once (safe, idempotent). If callers expect multi-step
            # forwarding, they may call this repeatedly; avoid spinning by bounding steps.
            max_steps = 8
            steps = 0
            while steps < max_steps:
                PhaseManager.next_phase(state, card_db)
                steps += 1
                # Stop early if we reach MAIN phase so active player can act
                if getattr(state, 'current_phase', None) == Phase.MAIN:
                    break
                # If pending effects appeared during transitions, stop
                if getattr(state, 'pending_effects', None):
                    break
        except Exception:
            # Be defensive in test/fallback environments
            pass

    @staticmethod
    def check_game_over(state: GameState, result_out: Any = None) -> tuple[bool, int]:
        return False, GameResult.NONE


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
    def resolve_action(state: GameState, action: Action, card_db: Any = None) -> None:
        try:
            gi = getattr(state, 'game_instance', None)
            if gi is not None and hasattr(gi, 'execute_action'):
                gi.execute_action(action)
            else:
                if action.type == ActionType.RESOLVE_EFFECT and state.pending_effects:
                    state.pending_effects.pop()
        except Exception:
            pass


class TensorConverter:
    @staticmethod
    def convert_to_tensor(state: Any, player_id: int, card_db: Any, mask_opponent: bool = True) -> List[float]:
        # Simulate C++ returning float vector
        return [0.0] * 856


__all__ = [
    'IS_NATIVE', 'GameInstance', 'GameState', 'Action', 'ActionType', 'PlayerIntent', 'ActionEncoder',
    'ActionGenerator', 'IntentGenerator', 'PhaseManager', 'EffectResolver', 'CardStub',
    'CardType', 'Phase', 'GameResult', 'GameCommand', 'CommandSystem', 'ExecuteActionCompat',
]

if CommandEncoder is not None:
    try:
        __all__.append('CommandEncoder')
    except Exception:
        __all__ = list(__all__) + ['CommandEncoder']

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
            try:
                if len(args) < 4:
                    return 0
                gs = args[0]
                key = args[1]

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

                try:
                    inst_lookup = _find_card_by_instance(int(key))
                except Exception:
                    inst_lookup = None

                if inst_lookup and (len(args) >= 4):
                    pid, from_zone_name, idx, card_obj = inst_lookup
                    target = args[3]
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
                        getattr(gs.players[pid], from_zone_name).pop(idx)
                    except Exception:
                        pass
                    try:
                        getattr(gs.players[pid], dst_attr).append(card_obj)
                    except Exception:
                        pass
                    return 1

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
                if not hasattr(state, 'hash_history'):
                    state.hash_history = []
                if not hasattr(state, 'calculate_hash'):
                    def _ch():
                        return 0
                    state.calculate_hash = _ch
                state.hash_history.append(getattr(state, 'calculate_hash')())
                state.hash_history.append(getattr(state, 'calculate_hash')())
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

    # Provide minimal, module-level fallback helpers (kept simple and robust)
    if 'ActionGenerator' not in globals():
        class ActionGenerator:
            def __init__(self, registry: Any = None):
                self.registry = registry

            def generate(self, state: Any, player_id: int) -> list:
                return []

    if 'ActionEncoder' not in globals():
        class ActionEncoder:
            def __init__(self):
                pass

            def encode(self, action: Any) -> dict:
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

    if 'TensorConverter' not in globals():
        class TensorConverter:
            @staticmethod
            def convert_to_tensor(state: Any, player_id: int, card_db: Any, mask_opponent: bool = True) -> List[float]:
                return [0.0] * 856

    if 'TokenConverter' not in globals():
        class TokenConverter:
            def to_tokens(self, obj: Any) -> list:
                try:
                    if obj is None:
                        return []
                    if isinstance(obj, list):
                        return [int(x) for x in obj][:256]
                    if hasattr(obj, 'instance_id'):
                        return [int(getattr(obj, 'instance_id')) % 8192]
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
                    return [abs(hash(str(obj))) % 8192]
                except Exception:
                    return []

            @staticmethod
            def get_vocab_size() -> int:
                return 8192

            @staticmethod
            def encode_state(state: Any, player_id: int, max_len: int = 512) -> list:
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
                try:
                    inst = state.get_card_instance(self.instance_id) if hasattr(state, 'get_card_instance') else None
                    if inst is None:
                        return
                    for p in state.players:
                        for zone_name in ('hand', 'battle_zone', 'mana_zone', 'shield_zone', 'graveyard', 'deck'):
                            z = getattr(p, zone_name, [])
                            for i, o in enumerate(list(z)):
                                if getattr(o, 'instance_id', None) == getattr(inst, 'instance_id', None):
                                    try:
                                        z.pop(i)
                                    except Exception:
                                        pass
                    dest = 'graveyard' if 'GRAVE' in str(self.to_zone).upper() else ('battle_zone' if 'BATTLE' in str(self.to_zone).upper() else 'hand')
                    try:
                        state.players[getattr(state, 'active_player_id', 0)].__dict__.setdefault(dest, []).append(inst)
                    except Exception:
                        pass
                except Exception:
                    pass

if 'DataCollector' not in globals():
    class DataCollector:
        def __init__(self, card_db: Any = None):
            self.card_db = card_db

        def collect_data_batch_heuristic(self, batch_size: int, include_history: bool, include_features: bool) -> Any:
            class Batch:
                def __init__(self):
                    self.values = []
            return Batch()

def get_card_stats(state: Any) -> Any:
    return {}


def index_to_command(action_index: int, state: Any, card_db: Any = None) -> dict:
    ACTION_MANA_SIZE = 20
    ACTION_PLAY_SIZE = 20
    MAX_BATTLE_SIZE = 20
    ACTION_BLOCK_SIZE = 20
    ACTION_SELECT_TARGET_SIZE = 100
    offset = 0

    if 0 <= action_index < offset + ACTION_MANA_SIZE:
        slot = action_index - offset
        return {'type': getattr(CommandType, 'MANA_CHARGE', 'MANA_CHARGE'), 'slot_index': slot}
    offset += ACTION_MANA_SIZE

    if offset <= action_index < offset + ACTION_PLAY_SIZE:
        slot = action_index - offset
        try:
            pid = getattr(state, 'active_player_id', 0)
            hand = list(getattr(state.players[pid], 'hand', []) or [])
            if 0 <= slot < len(hand):
                inst = getattr(hand[slot], 'instance_id', getattr(hand[slot], 'id', None))
                cid = getattr(hand[slot], 'card_id', getattr(hand[slot], 'id', None))
                return {'type': getattr(CommandType, 'PLAY_FROM_ZONE', 'PLAY_FROM_ZONE'), 'player': pid, 'slot_index': slot, 'instance_id': inst, 'card_id': cid, 'from_zone': 'hand', 'to_zone': 'battle_zone'}
        except Exception:
            pass
        return {'type': getattr(CommandType, 'PLAY_FROM_ZONE', 'PLAY_FROM_ZONE'), 'slot_index': slot, 'from_zone': 'hand', 'to_zone': 'battle_zone'}
    offset += ACTION_PLAY_SIZE

    attack_player_slots = MAX_BATTLE_SIZE
    attack_creature_slots = MAX_BATTLE_SIZE * MAX_BATTLE_SIZE

    if offset <= action_index < offset + attack_player_slots:
        slot = action_index - offset
        pid = getattr(state, 'active_player_id', 0)
        opp = 1 - pid
        try:
            battle = list(getattr(state.players[pid], 'battle_zone', []) or [])
            if 0 <= slot < len(battle):
                inst = getattr(battle[slot], 'instance_id', getattr(battle[slot], 'id', None))
                return {'type': getattr(CommandType, 'ATTACK', 'ATTACK'), 'source_instance_id': inst, 'target_player': opp}
        except Exception:
            pass
        return {'type': getattr(CommandType, 'ATTACK', 'ATTACK'), 'slot_index': slot, 'target_player': opp}
    offset += attack_player_slots

    if offset <= action_index < offset + attack_creature_slots:
        rel = action_index - offset
        atk_slot = rel // MAX_BATTLE_SIZE
        tgt_slot = rel % MAX_BATTLE_SIZE
        pid = getattr(state, 'active_player_id', 0)
        opp = 1 - pid
        try:
            atk_battle = list(getattr(state.players[pid], 'battle_zone', []) or [])
            def_battle = list(getattr(state.players[opp], 'battle_zone', []) or [])
            atk_inst = atk_battle[atk_slot] if 0 <= atk_slot < len(atk_battle) else None
            tgt_inst = def_battle[tgt_slot] if 0 <= tgt_slot < len(def_battle) else None
            atk_id = getattr(atk_inst, 'instance_id', getattr(atk_inst, 'id', None)) if atk_inst else None
            tgt_id = getattr(tgt_inst, 'instance_id', getattr(tgt_inst, 'id', None)) if tgt_inst else None
            return {'type': getattr(CommandType, 'ATTACK', 'ATTACK'), 'source_instance_id': atk_id, 'target_instance_id': tgt_id}
        except Exception:
            return {'type': getattr(CommandType, 'ATTACK', 'ATTACK'), 'slot_index': atk_slot, 'target_slot_index': tgt_slot}
    offset += attack_creature_slots

    if offset <= action_index < offset + ACTION_BLOCK_SIZE:
        slot = action_index - offset
        return {'type': getattr(CommandType, 'BLOCK', 'BLOCK'), 'slot_index': slot}
    offset += ACTION_BLOCK_SIZE

    if offset <= action_index < offset + ACTION_SELECT_TARGET_SIZE:
        slot = action_index - offset
        return {'type': getattr(CommandType, 'SELECT_TARGET', 'SELECT_TARGET'), 'target_index': slot}
    offset += ACTION_SELECT_TARGET_SIZE

    if action_index == offset:
        return {'type': getattr(CommandType, 'PASS', 'PASS')}
    offset += 1

    if action_index == offset:
        return {'type': getattr(CommandType, 'RESOLVE_EFFECT', 'RESOLVE_EFFECT')}
    offset += 1

    if action_index == offset:
        return {'type': getattr(CommandType, 'USE_SHIELD_TRIGGER', 'USE_SHIELD_TRIGGER')}
    return {'type': getattr(CommandType, 'NONE', 'NONE'), 'index': action_index}


def run_mcts_and_get_command(root_state: Any, onnx_path: str, **kwargs: Any) -> dict:
    if 'run_mcts_with_onnx' not in globals():
        raise ImportError('run_mcts_with_onnx not found; ensure dm_ai_module updated')
    res = run_mcts_with_onnx(root_state, onnx_path, **kwargs)
    cmd = None
    try:
        idx = res.get('best_action_index', None)
        if idx is not None:
            cmd = index_to_command(int(idx), root_state)
    except Exception:
        cmd = None
    res['best_action_command'] = cmd
    return res


def apply_command(state: Any, command: dict, source_id: int = -1, player_id: Optional[int] = None, ctx: Any = None) -> bool:
    try:
        if player_id is None:
            player_id = getattr(state, 'active_player_id', 0)

        if hasattr(command, 'execute') and hasattr(command, 'type'):
            try:
                CommandSystem.execute_command(state, command, source_id, player_id, ctx)
                return True
            except Exception:
                return False

        cmd = GameCommand()
        try:
            cmd.type = command.get('type', getattr(cmd, 'type', None))
        except Exception:
            pass
        try:
            if 'source_instance_id' in command:
                cmd.source_instance_id = command.get('source_instance_id')
            elif 'instance_id' in command:
                cmd.source_instance_id = command.get('instance_id')
        except Exception:
            pass
        try:
            if 'target_instance_id' in command:
                cmd.target_instance_id = command.get('target_instance_id')
            if 'target_player' in command:
                cmd.target_player = command.get('target_player')
        except Exception:
            pass
        try:
            if 'card_id' in command:
                cmd.card_id = command.get('card_id')
        except Exception:
            pass

        try:
            CommandSystem.execute_command(state, cmd, int(cmd.source_instance_id) if getattr(cmd, 'source_instance_id', -1) is not None else source_id, player_id, ctx)
            return True
        except Exception:
            return False
    except Exception:
        return False


def commands_from_actions(actions: list, state: Optional[Any] = None) -> list:
    out = []
    for a in (actions or []):
        try:
            if a is None:
                continue
            if isinstance(a, dict):
                out.append(a)
                continue
            cmd = getattr(a, 'command', None)
            if cmd:
                out.append(cmd)
                continue
            cdict = {}
            try:
                cdict['type'] = getattr(a, 'type', None) or getattr(a, 'action_type', None)
            except Exception:
                pass
            try:
                if hasattr(a, 'source_instance_id'):
                    cdict['source_instance_id'] = getattr(a, 'source_instance_id')
                elif hasattr(a, 'instance_id'):
                    cdict['source_instance_id'] = getattr(a, 'instance_id')
            except Exception:
                pass
            try:
                if hasattr(a, 'target_player'):
                    cdict['target_player'] = getattr(a, 'target_player')
                if hasattr(a, 'target_instance_id'):
                    cdict['target_instance_id'] = getattr(a, 'target_instance_id')
            except Exception:
                pass
            try:
                if hasattr(a, 'card_id'):
                    cdict['card_id'] = getattr(a, 'card_id')
            except Exception:
                pass
            out.append(cdict)
        except Exception:
            continue
    return out


# If we loaded the native extension earlier, augment it with any Python-only
# helper symbols defined in this shim so tests and tooling can import a
# superset of native functionality (e.g., `CardStub` helpers).
try:
    if IS_NATIVE:
        import sys as _sys
        _native = _sys.modules.get('dm_ai_module')
        if _native is not None and isinstance(_native, type(importlib)) or True:
            for _name, _obj in list(globals().items()):
                if _name.startswith('_'):
                    continue
                if _name in ('sys','os','importlib','importlib_util','importlib_machinery'):
                    continue
                try:
                    if not hasattr(_native, _name):
                        setattr(_native, _name, _obj)
                except Exception:
                    continue
            try:
                setattr(_native, 'IS_NATIVE', True)
            except Exception:
                pass
except Exception:
    pass


def generate_commands(state: Any, card_db: Any = None) -> list:
    actions = []
    try:
        if 'ActionGenerator' in globals() and hasattr(ActionGenerator, 'generate_legal_actions'):
            try:
                actions = ActionGenerator.generate_legal_actions(state, card_db)
            except Exception:
                try:
                    actions = ActionGenerator().generate(state, getattr(state, 'active_player_id', 0))
                except Exception:
                    actions = []
        else:
            actions = []
    except Exception:
        actions = []

    return commands_from_actions(actions, state)

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


if 'JsonLoader' not in globals():
    class JsonLoader:
        @staticmethod
        def load_cards(filepath: str) -> dict[int, Any]:
            final = filepath
            if not os.path.exists(final):
                alt = os.path.join(os.path.dirname(__file__), filepath)
                if os.path.exists(alt):
                    final = alt
            try:
                with open(final, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    out: dict[int, Any] = {}
                    for item in data:
                        try:
                            out[int(item.get('id'))] = item
                        except Exception:
                            continue
                    return out
                if isinstance(data, dict):
                    out: dict[int, Any] = {}
                    for k, v in data.items():
                        try:
                            out[int(k)] = v
                        except Exception:
                            try:
                                out[int(v.get('id'))] = v
                            except Exception:
                                continue
                    return out
            except Exception:
                return {}
            return {}

if 'MCTS' not in globals():
    class MCTSNode:
        def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action: Any = None) -> None:
            self.state = state
            self.parent = parent
            self.action = action
            self.children: List['MCTSNode'] = []
            self.visit_count = 0
            self.value_sum = 0.0
            self.prior = 0.0

        def is_expanded(self) -> bool:
            return len(self.children) > 0

        def value(self) -> float:
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count

    class MCTS:
        def __init__(self, network: Any, card_db: Any, simulations: int = 100, c_puct: float = 1.0,
                     dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                     state_converter: Any = None, action_encoder: Any = None) -> None:
            self.network = network
            self.card_db = card_db
            self.simulations = simulations
            self.c_puct = c_puct
            self.dirichlet_alpha = dirichlet_alpha
            self.dirichlet_epsilon = dirichlet_epsilon
            self.state_converter = state_converter
            self.action_encoder = action_encoder

        def _fast_forward(self, state: Any) -> None:
            PhaseManager.fast_forward(state, self.card_db)

        def search(self, root_state: Any, add_noise: bool = False) -> MCTSNode:
            root_state_clone = root_state.clone()
            self._fast_forward(root_state_clone)
            root = MCTSNode(root_state_clone)

            self._expand(root)

            if add_noise and 'np' in globals():
                self._add_exploration_noise(root)

            for _ in range(self.simulations):
                node = root
                while node.is_expanded():
                    next_node = self._select_child(node)
                    if next_node is None:
                        break
                    node = next_node

                if not node.is_expanded():
                    value = self._expand(node)
                else:
                    value = node.value()

                self._backpropagate(node, value)

            return root

        def _add_exploration_noise(self, node: MCTSNode) -> None:
            if not node.children:
                return
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
            for i, child in enumerate(node.children):
                child.prior = child.prior * (1 - self.dirichlet_epsilon) + noise[i] * self.dirichlet_epsilon

        def _select_child(self, node: MCTSNode) -> Optional[MCTSNode]:
            best_score = -float('inf')
            best_child = None
            for child in node.children:
                u_score = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
                q_score = child.value()
                score = q_score + u_score
                if best_score < score:
                    best_score = score
                    best_child = child
            if best_child is None and node.children:
                best_child = node.children[0]
            return best_child

        def _expand(self, node: MCTSNode) -> float:
            is_over, result = PhaseManager.check_game_over(node.state)
            if is_over:
                current_player = getattr(node.state, 'active_player_id', 0)
                if result == GameResult.DRAW: return 0.0
                if result == GameResult.P1_WIN: return 1.0 if current_player == 0 else -1.0
                if result == GameResult.P2_WIN: return 1.0 if current_player == 1 else -1.0
                return 0.0

            actions = ActionGenerator.generate_legal_actions(node.state, self.card_db)
            if not actions:
                return 0.0

            # Encode State
            tensor_t = None
            if self.state_converter:
                tensor = self.state_converter(node.state, getattr(node.state, 'active_player_id', 0), self.card_db)
                if 'torch' in globals():
                    if isinstance(tensor, torch.Tensor):
                        tensor_t = tensor
                        if tensor_t.dim() == 1: tensor_t = tensor_t.unsqueeze(0)
                    elif isinstance(tensor, (list, np.ndarray)):
                        # FIX: Handle Int Tokens for Transformer
                        is_int = False
                        if isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], int):
                            is_int = True
                        elif isinstance(tensor, np.ndarray) and np.issubdtype(tensor.dtype, np.integer):
                            is_int = True

                        dtype = torch.long if is_int else torch.float32
                        tensor_t = torch.tensor(tensor, dtype=dtype).unsqueeze(0)

            if 'torch' in globals() and tensor_t is not None:
                with torch.no_grad():
                     policy_logits, value = self.network(tensor_t)
                policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
                val = float(value.item())
            else:
                policy = [1.0/len(actions)] * 1024
                val = 0.0

            # Create children
            for i, act in enumerate(actions):
                # Simple prior mapping (assuming actions match policy index roughly or uniform)
                # In real engine, we need ActionEncoder. Here just take uniform or index match
                idx = i % len(policy) if len(policy) > 0 else 0
                prior = float(policy[idx])

                next_state = node.state.clone()
                # Execute
                if hasattr(next_state, 'execute_action'): # If GameInstance linked
                     next_state.execute_action(act)
                else:
                     # Simulate execution via PhaseManager/CommandSystem logic
                     if act.type == ActionType.PASS:
                         PhaseManager.next_phase(next_state, self.card_db)
                     elif act.type == ActionType.MANA_CHARGE:
                         pid = getattr(next_state, 'active_player_id', 0)
                         next_state.players[pid].mana_zone.append(CardStub(getattr(act, 'card_id', 0)))
                         # Remove from hand?
                         hand = next_state.players[pid].hand
                         if hand: hand.pop(0)

                child = MCTSNode(next_state, parent=node, action=act)
                child.prior = prior
                node.children.append(child)

            return val

        def _backpropagate(self, node: Optional[MCTSNode], value: float) -> None:
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = -value
                node = node.parent

# Backwards-compatible fallbacks: ensure common public symbols exist
if 'GameInstance' not in globals():
    class GameInstance:
        def __init__(self, seed: int = 0, card_db: Any = None):
            self.state = GameState()
            self.card_db = card_db

        def start_game(self):
            try:
                PhaseManager.start_game(self.state, self.card_db)
            except Exception:
                try:
                    self.state.setup_test_duel()
                except Exception:
                    pass

        def execute_action(self, action: Any) -> None:
            # Minimal delegate to GameInstance-like behavior on state if possible
            try:
                if hasattr(self.state, 'game_instance') and hasattr(self.state.game_instance, 'execute_action'):
                    return self.state.game_instance.execute_action(action)
            except Exception:
                pass

if 'CommandSystem' not in globals():
    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, source_id: int = -1, player_id: int = 0, ctx: Any = None) -> None:
            try:
                # Prefer any previously defined execute_command
                existing = globals().get('CommandSystem')
                if existing and existing is not CommandSystem and hasattr(existing, 'execute_command'):
                    return existing.execute_command(state, cmd, source_id, player_id, ctx)
            except Exception:
                pass
            try:
                if hasattr(cmd, 'execute'):
                    return cmd.execute(state)
            except Exception:
                pass

if 'CardStub' not in globals():
    class CardStub:
        def __init__(self, card_id: int, instance_id: Optional[int] = None):
            self.card_id = card_id
            self.instance_id = instance_id if instance_id is not None else 0
            self.is_tapped = False
            self.sick = False


# Compatibility helper: prefer command-first execution, fall back to legacy execute_action
def ExecuteActionCompat(target: Any, action: Any, player_id: int = 0, ctx: Any = None) -> bool:
    """Execute an action using command-first semantics when possible.

    - If `action` is a dict-like command, prefer `CommandSystem.execute_command` or
      `EngineCompat.ExecuteCommand` (if available).
    - If `target` exposes `execute_action`, call it as a last resort.
    Returns True on success, False otherwise.
    """
    try:
        # If action already exposes execute(), try that first
        if hasattr(action, 'execute') and callable(getattr(action, 'execute')):
            try:
                action.execute(getattr(target, 'state', target))
                return True
            except Exception:
                pass

        # If action is dict-like, prefer command execution APIs
        if isinstance(action, dict):
            # Try CommandSystem if present
            try:
                if 'CommandSystem' in globals() and hasattr(CommandSystem, 'execute_command'):
                    CommandSystem.execute_command(getattr(target, 'state', target), action, -1, player_id, ctx)
                    return True
            except Exception:
                pass

            # EngineCompat.ExecuteCommand fallback (used by toolkit)
            try:
                from dm_toolkit.engine.compat import EngineCompat
                EngineCompat.ExecuteCommand(getattr(target, 'state', target), action, None)
                return True
            except Exception:
                pass

        # If target is a GameInstance-like object, call its execute_action
        try:
            if hasattr(target, 'execute_action') and callable(getattr(target, 'execute_action')):
                target.execute_action(action)
                return True
        except Exception:
            pass

        # As last resort, if target is a plain state, try CommandSystem on it
        try:
            if 'CommandSystem' in globals() and hasattr(CommandSystem, 'execute_command'):
                CommandSystem.execute_command(target, action, -1, player_id, ctx)
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False


# Ensure core symbols exist on the module object even if a native extension
# was loaded earlier and didn't export them. This makes `from dm_ai_module import X`
# robust during incremental migration.
try:
    import sys as _sys
    _mod = _sys.modules.get('dm_ai_module')
    if _mod is not None:
        if not hasattr(_mod, 'GameInstance') and 'GameInstance' in globals():
            setattr(_mod, 'GameInstance', globals().get('GameInstance'))
        if not hasattr(_mod, 'GameResult') and 'GameResult' in globals():
            setattr(_mod, 'GameResult', globals().get('GameResult'))
        if not hasattr(_mod, 'ExecuteActionCompat'):
            setattr(_mod, 'ExecuteActionCompat', ExecuteActionCompat)
except Exception:
    pass
