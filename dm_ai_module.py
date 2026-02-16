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

                if tt == 'PASS' or t == 'PASS' or t == CommandType.PASS:
                    return 0
                if tt == 'MANA_CHARGE' or t == 'MANA_CHARGE' or t == CommandType.MANA_CHARGE:
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

            if tt == 'PASS' or t == 'PASS' or t == CommandType.PASS:
                return 0

            slot_candidates = []
            if isinstance(cmd, dict):
                for k in ('slot_index', 'instance_id', 'source_instance_id', 'id'):
                    if k in cmd:
                        slot_candidates.append(cmd.get(k))
            else:
                for k in ('slot_index', 'instance_id', 'source_instance_id', 'id'):
                    slot_candidates.append(getattr(cmd, k, None))

            if tt == 'MANA_CHARGE' or t == 'MANA_CHARGE' or t == CommandType.MANA_CHARGE:
                for sc in slot_candidates:
                    if sc is None:
                        continue
                    try:
                        si = int(sc)
                    except Exception:
                        continue
                    if si < CommandEncoder.MANA_CHARGE_BASE or si >= CommandEncoder.PLAY_FROM_ZONE_BASE:
                        si = CommandEncoder.MANA_CHARGE_BASE + (abs(si) % CommandEncoder.MANA_CHARGE_SLOTS)
                    return si
                try:
                    s = json.dumps(cmd, sort_keys=True)
                except Exception:
                    s = str(cmd)
                return CommandEncoder.MANA_CHARGE_BASE + (abs(hash(s)) % CommandEncoder.MANA_CHARGE_SLOTS)

            if tt == 'PLAY_FROM_ZONE' or tt == 'PLAY' or t in ('PLAY_FROM_ZONE', 'PLAY') or t == CommandType.PLAY_FROM_ZONE:
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

            if tt == 'ATTACK' or t == 'ATTACK' or t == CommandType.ATTACK_PLAYER or t == CommandType.ATTACK_CREATURE:
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

            try:
                s = json.dumps(cmd, sort_keys=True)
            except Exception:
                s = str(cmd)
            try:
                return abs(hash(s)) % CommandEncoder.TOTAL_COMMAND_SIZE
            except Exception:
                return 0


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
                if t in (CommandType.MANA_CHARGE, 'MANA_CHARGE'):
                    pid = getattr(state, 'active_player_id', player_id)
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
                    if not hasattr(state, 'command_history'):
                        state.command_history = []
                    state.command_history.append(cmd)
                elif t == 'DRAW_CARD' or t == getattr(CommandType, 'DRAW_CARD', 'DRAW_CARD'):
                    # Simulated minimal DRAW_CARD logic for testing interactive flow
                    if cmd.get('up_to') or cmd.get('upto'):
                        # Simulate WAIT_INPUT
                        state.waiting_for_user_input = True
                        class _Query:
                            def __init__(self, qt, p):
                                self.query_type = qt
                                self.params = p
                        state.pending_query = _Query("SELECT_NUMBER", {'min': 0, 'max': cmd.get('amount', 1)})
                        return # Pause execution

                    # Auto draw (simplified)
                    amt = int(cmd.get('amount', 1) or 1)
                    pid = player_id
                    p = state.players[pid]
                    for _ in range(amt):
                         if p.deck:
                             p.hand.append(p.deck.pop())
                    if not hasattr(state, 'command_history'):
                        state.command_history = []
                    state.command_history.append(cmd)
                elif t == 'DESTROY' or t == getattr(CommandType, 'DESTROY', 'DESTROY'):
                    if not hasattr(state, 'execution_context'):
                        return
                    ctx_vars = getattr(state.execution_context, 'variables', {})
                    input_key = cmd.get('input_value_key') or 'selected'
                    target_ids = ctx_vars.get(input_key, [])
                    if not isinstance(target_ids, list):
                        target_ids = [target_ids]
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
                elif t == 'SELECT_TARGET' or t == getattr(CommandType, 'SELECT_TARGET', 'SELECT_TARGET'):
                    if not hasattr(state, 'execution_context'):
                        class _EC:
                            def __init__(self):
                                self.variables = {}
                        state.execution_context = _EC()
                    ctx_vars = getattr(state.execution_context, 'variables', {})
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

    def calculate_hash(self) -> int:
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
        try:
            if hasattr(cmd, '__dict__'):
                d = cmd.__dict__
            else:
                d = cmd
            if 'CommandSystem' in globals():
                CommandSystem.execute_command(self, d)
        except Exception:
            pass

    def make_move(self, cmd: Any) -> None:
        try:
            self._last_snap = self.create_snapshot()
            if not hasattr(self, 'command_history'):
                self.command_history = []
            self.command_history.append(cmd)
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
        if not hasattr(self, '_next_instance_id'):
            self._next_instance_id = 1000
        self._next_instance_id += 1
        return int(self._next_instance_id)

    def setup_test_duel(self):
        self.players = [Player(0), Player(1)]
        for p in self.players:
            p.hand.clear()
            p.mana_zone.clear()
            p.battle_zone.clear()
            p.shield_zone.clear()
            p.graveyard.clear()
            p.deck.clear()
        self.turn_number = 1
        self.active_player_id = 0
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
        if 0 <= player_id < len(self.player_modes):
            return self.player_modes[player_id] == PlayerMode.HUMAN
        return False

    def add_card_to_hand(self, player: int, card_id: int, instance_id: Optional[int] = None, count: int = 1):
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
        try:
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
        _Phase = globals().get('Phase')
        if _Phase is not None:
            self.state.current_phase = _Phase.MANA
        else:
            self.state.current_phase = 2
        self.state.active_player_id = 0

    def initialize_card_stats(self, deck_size: int):
        pass

    def execute_command(self, cmd):
        """Execute a game command (CommandDef or dict)."""
        try:
            if hasattr(self.state, 'apply_move'):
                try:
                    self.state.apply_move(cmd)
                    return
                except Exception:
                    pass

            _CommandSystem = globals().get('CommandSystem')
            if _CommandSystem is not None:
                try:
                    _CommandSystem.execute_command(
                        self.state, 
                        cmd,
                        source_id=getattr(cmd, 'instance_id', -1),
                        player_id=getattr(self.state, 'active_player_id', 0)
                    )
                    return
                except Exception:
                    pass
        except Exception:
            pass

    def step(self) -> bool:
        if getattr(self.state, 'game_over', False):
            return False
        
        try:
            cmds = IntentGenerator.generate_legal_commands(self.state, self.card_db)
        except Exception:
            cmds = []
        
        if not cmds:
            return False
        
        non_pass_cmds = []
        for cmd in cmds:
            cmd_type = cmd.get('type') if isinstance(cmd, dict) else getattr(cmd, 'type', None)
            try:
                if cmd_type != CommandType.PASS and cmd_type != 'PASS':
                    non_pass_cmds.append(cmd)
            except Exception:
                if cmd_type != 'PASS':
                    non_pass_cmds.append(cmd)
        
        cmd_to_execute = non_pass_cmds[0] if non_pass_cmds else cmds[0]
        
        try:
            self.execute_command(cmd_to_execute)
            PhaseManager.fast_forward(self.state, self.card_db)
            return True
        except Exception:
            return False

    def resolve_command(self, cmd: Any) -> None:
        try:
            self.execute_command(cmd)
        except Exception:
            pass


class ActionEncoder:
    @staticmethod
    def action_to_index(action: Any) -> int:
        try:
            if 'CommandEncoder' in globals() and CommandEncoder is not None:
                try:
                    if isinstance(action, dict):
                        return CommandEncoder.command_to_index(action)
                    t = getattr(action, 'type', None)
                    if t is not None:
                        cmd = {}
                        if t == CommandType.PASS:
                            cmd['type'] = 'PASS'
                        elif t == CommandType.MANA_CHARGE:
                            cmd['type'] = 'MANA_CHARGE'
                            cmd['slot_index'] = getattr(action, 'source_instance_id', 1) or 1
                        elif t == CommandType.PLAY_FROM_ZONE:
                            cmd['type'] = 'PLAY_FROM_ZONE'
                            cmd['slot_index'] = getattr(action, 'source_instance_id', 0) or 0
                        else:
                            raise ValueError('no command mapping')
                        return CommandEncoder.command_to_index(cmd)
                except Exception:
                    pass

            key = (getattr(action, 'type', 0), getattr(action, 'card_id', -1), getattr(action, 'source_instance_id', -1))
            return abs(hash(key)) % 1024
        except Exception:
            return -1


class IntentGenerator:
    @staticmethod
    def generate_legal_commands(state: GameState, card_db: Any = None) -> List[Any]:
        out = []
        try:
            pid = getattr(state, 'active_player_id', 0)
            p = state.players[pid]

            # PASS is always legal
            cmd = {'type': 'PASS', 'uid': str(uuid.uuid4())}
            out.append(cmd)

            phase = getattr(state, 'current_phase', Phase.MANA)

            if phase == Phase.MANA:
                for c in list(p.hand):
                    m = {'type': 'MANA_CHARGE', 'instance_id': c.instance_id, 'card_id': c.card_id, 'uid': str(uuid.uuid4())}
                    out.append(m)

            elif phase == Phase.MAIN:
                for c in list(p.hand):
                    # Simplified play logic
                    pc = {'type': 'PLAY_FROM_ZONE', 'instance_id': c.instance_id, 'card_id': c.card_id, 'uid': str(uuid.uuid4())}
                    out.append(pc)

            elif phase == Phase.ATTACK:
                for c in list(p.battle_zone):
                     if not c.is_tapped and not c.sick:
                        at = {'type': 'ATTACK', 'instance_id': c.instance_id, 'target_player': 1 - pid, 'uid': str(uuid.uuid4())}
                        out.append(at)

        except Exception:
            pass
        return out

ActionGenerator = IntentGenerator


class PhaseManager:
    @staticmethod
    def start_game(state: GameState, card_db: Any = None) -> None:
        try:
            state.current_phase = Phase.MANA
            state.active_player_id = 0
            for pid in (0, 1):
                p = state.players[pid]
                shields_to_place = min(5, len(p.deck))
                for _ in range(shields_to_place):
                    if p.deck:
                        card = p.deck.pop()
                        p.shield_zone.append(card)
                cards_to_draw = min(5, len(p.deck))
                for _ in range(cards_to_draw):
                    if p.deck:
                        card = p.deck.pop()
                        p.hand.append(card)
        except Exception:
            pass

    @staticmethod
    def setup_scenario(state: GameState, config: Any, card_db: Any = None) -> None:
        pass

    @staticmethod
    def next_phase(state: GameState, card_db: Any = None) -> None:
        try:
            def _phase_value(x):
                try: return int(x)
                except: pass
                try: return int(getattr(x, 'value'))
                except: pass
                try: return int(str(x))
                except: return 5

            cur = _phase_value(state.current_phase)
            _Phase = globals().get('Phase')
            if _Phase is None:
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
                state.active_player_id = 1 - state.active_player_id
                state.current_phase = _Phase.MANA
                p = state.players[state.active_player_id]
                for c in p.mana_zone:
                    c.is_tapped = False
                for c in p.battle_zone:
                    c.is_tapped = False
                    c.sick = False
                p = state.players[state.active_player_id]
                if p.deck:
                    card = p.deck.pop()
                    p.hand.append(card)
                if state.active_player_id == 0:
                    state.turn_number += 1
        except Exception:
            pass

    @staticmethod
    def fast_forward(state: GameState, card_db: Any = None) -> None:
        try:
            cp = getattr(state, 'current_phase', None)
            try:
                if not isinstance(cp, Phase):
                    state.current_phase = Phase(cp)
            except Exception:
                state.current_phase = Phase.MANA

            if getattr(state, 'pending_effects', None):
                return

            max_steps = 8
            steps = 0
            while steps < max_steps:
                PhaseManager.next_phase(state, card_db)
                steps += 1
                if getattr(state, 'current_phase', None) == Phase.MAIN:
                    break
                if getattr(state, 'pending_effects', None):
                    break
        except Exception:
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
    def resolve_action(state: GameState, action: Any, card_db: Any = None) -> None:
        pass


class TensorConverter:
    @staticmethod
    def convert_to_tensor(state: Any, player_id: int, card_db: Any, mask_opponent: bool = True) -> List[float]:
        return [0.0] * 856


__all__ = [
    'IS_NATIVE', 'GameInstance', 'GameState', 'CommandType', 'ActionEncoder',
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
            return 0
        @staticmethod
        def trigger_loop_detection(state: Any):
            pass

if 'ParallelRunner' not in globals():
    class ParallelRunner:
        def __init__(self, card_db: Any, sims: int, batch_size: int):
            self.card_db = card_db
            self.sims = sims
            self.batch_size = batch_size

        def play_games(self, initial_states: List[Any], evaluator_func: Any, temperature: float, add_noise: bool, threads: int) -> List[Any]:
            return []

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
                return []

            @staticmethod
            def get_vocab_size() -> int:
                return 8192

            @staticmethod
            def encode_state(state: Any, player_id: int, max_len: int = 512) -> list:
                return []

    if 'TransitionCommand' not in globals():
        class TransitionCommand:
            def __init__(self, instance_id: int = -1, from_zone: str = '', to_zone: str = '', **kwargs: Any):
                pass
            def execute(self, state: Any) -> None:
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
    return {'type': 'PASS', 'index': action_index}


def run_mcts_and_get_command(root_state: Any, onnx_path: str, **kwargs: Any) -> dict:
    return {}


def apply_command(state: Any, command: dict, source_id: int = -1, player_id: Optional[int] = None, ctx: Any = None) -> bool:
    try:
        if hasattr(state, 'apply_move'):
            state.apply_move(command)
            return True
        return False
    except Exception:
        return False


def commands_from_actions(actions: list, state: Optional[Any] = None) -> list:
    return actions


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
    return IntentGenerator.generate_legal_commands(state, card_db)

if 'DeckEvolutionConfig' not in globals():
    class DeckEvolutionConfig:
        def __init__(self):
            pass

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
             pass


if 'JsonLoader' not in globals():
    class JsonLoader:
        @staticmethod
        def load_cards(filepath: str) -> dict[int, Any]:
            return {}

if 'MCTS' not in globals():
    class MCTSNode:
        def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, action: Any = None) -> None:
            pass

    class MCTS:
        def __init__(self, network: Any, card_db: Any, simulations: int = 100, c_puct: float = 1.0,
                     dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                     state_converter: Any = None, action_encoder: Any = None) -> None:
            pass

        def search(self, root_state: Any, add_noise: bool = False) -> Any:
            return None

# Backwards-compatible fallbacks: ensure common public symbols exist
if 'GameInstance' not in globals():
    class GameInstance:
        pass

if 'CommandSystem' not in globals():
    class CommandSystem:
        @staticmethod
        def execute_command(state: Any, cmd: Any, source_id: int = -1, player_id: int = 0, ctx: Any = None) -> None:
            pass

if 'CardStub' not in globals():
    class CardStub:
        pass


# Compatibility helper: prefer command-first execution, fall back to legacy execute_action
def ExecuteActionCompat(target: Any, action: Any, player_id: int = 0, ctx: Any = None) -> bool:
    try:
        if hasattr(target, 'execute_command'):
            target.execute_command(action)
            return True
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
