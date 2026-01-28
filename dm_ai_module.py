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

try:
    import torch
    import numpy as np
except ImportError:
    pass

# Indicate native extension is not loaded
IS_NATIVE = False


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


class CommandType(IntEnum):
    NONE = 0
    PLAY_FROM_ZONE = 1
    MANA_CHARGE = 2
    TRANSITION = 3
    ATTACK = 4
    PASS = 5


class CommandSystem:
    @staticmethod
    def execute_command(state: Any, cmd: Any, source_id: int = -1, player_id: int = 0, ctx: Any = None) -> None:
        # Minimal wrapper: attempt to call `execute` on cmd or treat as Action
        try:
            if hasattr(cmd, 'execute') and callable(getattr(cmd, 'execute')):
                try:
                    cmd.execute(state)
                    return
                except TypeError:
                    cmd.execute(state, ctx)
            # Fallback: if cmd is dict-like, try to map basic commands
            if isinstance(cmd, dict):
                t = cmd.get('type')
                if t in (CommandType.MANA_CHARGE, 'MANA_CHARGE'):
                    # emulate mana charge
                    pid = getattr(state, 'active_player_id', player_id)
                    cid = cmd.get('card_id') or cmd.get('instance_id') or 0
                    state.players[pid].mana_zone.append(CardStub(cid))
        except Exception:
            pass


class CardType(IntEnum):
    CREATURE = 0
    SPELL = 1


class Phase(IntEnum):
    MANA = 2
    MAIN = 3
    ATTACK = 4
    END = 5


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
    def __init__(self):
        self.players: List[Player] = [Player(0), Player(1)]
        self.current_phase = Phase.MANA
        self.active_player_id = 0
        self.pending_effects: List[Any] = []
        self.turn_number = 1
        self.game_over = False
        self.winner = -1

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
        try:
            self.players[player].deck = list(deck_ids)
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

    def clone(self):
        return copy.deepcopy(self)


class GameInstance:
    def __init__(self, seed: int = 0, card_db: Any = None):
        self.state = GameState()
        self.card_db = card_db

    def start_game(self):
        self.state.current_phase = Phase.MANA
        self.state.active_player_id = 0

    def initialize_card_stats(self, deck_size: int):
        pass

    def execute_action(self, action: Action):
        if action.type == ActionType.PLAY_CARD or action.type == ActionType.DECLARE_PLAY:
            pid = getattr(action, 'target_player', getattr(self.state, 'active_player_id', 0))
            hand = self.state.players[pid].hand
            found = None
            for c in list(hand):
                if getattr(c, 'card_id', None) == getattr(action, 'card_id', None) or getattr(c, 'instance_id', None) == getattr(action, 'source_instance_id', None):
                    found = c
                    try:
                        hand.remove(c)
                    except Exception:
                        pass
                    break
            eff = type('Eff', (), {})()
            try:
                eff.card_id = action.card_id
            except Exception:
                eff.card_id = None
            self.state.pending_effects.append(eff)
        elif action.type == ActionType.RESOLVE_EFFECT:
            if self.state.pending_effects:
                eff = self.state.pending_effects.pop()
                pid = getattr(self.state, 'active_player_id', 0)
                try:
                    cid = getattr(eff, 'card_id', None)
                    if cid is not None:
                        self.state.players[pid].graveyard.append(CardStub(cid))
                except Exception:
                    pass
        elif action.type == ActionType.MANA_CHARGE:
            pid = getattr(self.state, 'active_player_id', 0)
            cid = getattr(action, 'card_id', None)
            self.state.players[pid].mana_zone.append(CardStub(cid if cid is not None else 0))


class ActionEncoder:
    @staticmethod
    def action_to_index(action: Any) -> int:
        try:
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


class IntentGenerator(ActionGenerator):
    pass


class PhaseManager:
    @staticmethod
    def start_game(state: GameState, card_db: Any = None) -> None:
        try:
            state.current_phase = Phase.MANA
            state.active_player_id = 0
        except Exception:
            pass

    @staticmethod
    def next_phase(state: GameState, card_db: Any = None) -> None:
        try:
            if state.current_phase == Phase.MANA:
                state.current_phase = Phase.MAIN
            elif state.current_phase == Phase.MAIN:
                state.current_phase = Phase.ATTACK
            elif state.current_phase == Phase.ATTACK:
                state.current_phase = Phase.END
            else:
                # END -> Next Turn (MANA)
                state.active_player_id = 1 - state.active_player_id
                state.current_phase = Phase.MANA

                # Untap Step
                p = state.players[state.active_player_id]
                for c in p.mana_zone:
                    c.is_tapped = False
                for c in p.battle_zone:
                    c.is_tapped = False
                    c.sick = False # Remove summoning sickness

                # Draw Step
                # Use a placeholder card ID for draw (use 1 to be safe within vocab limits)
                state.add_card_to_hand(state.active_player_id, 1)

                # Increment Turn Counter (assuming increment on P0 start or every turn?)
                # Usually standard practice:
                if state.active_player_id == 0:
                    state.turn_number += 1
        except Exception:
            pass

    @staticmethod
    def fast_forward(state: GameState, card_db: Any = None) -> None:
        return

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
    'IS_NATIVE', 'GameInstance', 'GameState', 'Action', 'ActionType', 'ActionEncoder',
    'ActionGenerator', 'IntentGenerator', 'PhaseManager', 'EffectResolver', 'CardStub',
    'CardType', 'Phase', 'GameResult', 'GameCommand',
]

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
                    # Simulate C++ returning float vector
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
        def execute(self, state: Any) -> None:
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

            if tensor_t is None:
                 return 0.0

            if 'torch' in globals():
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
