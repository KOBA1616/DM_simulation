import json
from types import SimpleNamespace

class JsonLoader:
    @staticmethod
    def load_cards(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        out = {}
        for c in data:
            ns = SimpleNamespace()
            ns.name = c.get('name')
            ns.cost = c.get('cost')
            ns.power = c.get('power')
            civs = c.get('civilizations', [])
            ns.civilizations = [Civilization[v] if isinstance(v, str) and v in Civilization.__members__ else v for v in civs]
            t = c.get('type')
            ns.type = CardType[t] if isinstance(t, str) and t in CardType.__members__ else t
            # expose keywords as attribute-accessible object
            ns.keywords = SimpleNamespace(**c.get('keywords', {}))
            ns.effects = c.get('effects', [])
            out[c.get('id')] = ns
        return out

from enum import Enum, IntEnum

class Civilization(Enum):
    FIRE = 'FIRE'

class CardType(Enum):
    CREATURE = 'CREATURE'


# --- Lightweight test stubs for bindings expected by unit tests ---
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional


class Zone(Enum):
    HAND = 'HAND'
    DECK = 'DECK'
    MANA_ZONE = 'MANA_ZONE'
    GRAVEYARD = 'GRAVEYARD'
    BATTLE = 'BATTLE'
    # alias expected by some tests
    MANA = 'MANA'


class Phase(IntEnum):
    MAIN = 0
    ATTACK = 1


class CommandType(Enum):
    TRANSITION = 'TRANSITION'
    MUTATE = 'MUTATE'
    FLOW = 'FLOW'
    QUERY = 'QUERY'
    DRAW_CARD = 'DRAW_CARD'
    DISCARD = 'DISCARD'
    DESTROY = 'DESTROY'
    MANA_CHARGE = 'MANA_CHARGE'
    TAP = 'TAP'
    UNTAP = 'UNTAP'
    POWER_MOD = 'POWER_MOD'
    ADD_KEYWORD = 'ADD_KEYWORD'
    RETURN_TO_HAND = 'RETURN_TO_HAND'
    BREAK_SHIELD = 'BREAK_SHIELD'
    SEARCH_DECK = 'SEARCH_DECK'
    SHIELD_TRIGGER = 'SHIELD_TRIGGER'
    NONE = 'NONE'


def _zone_name(zone: Any) -> str:
    # Accept Zone enum or string
    if isinstance(zone, Zone):
        if zone in (Zone.MANA, Zone.MANA_ZONE):
            return 'mana_zone'
        if zone == Zone.HAND:
            return 'hand'
        if zone == Zone.BATTLE:
            return 'battle'
        if zone == Zone.DECK:
            return 'deck'
    if isinstance(zone, str):
        s = zone.upper()
        if 'MANA' in s:
            return 'mana_zone'
        if 'HAND' in s:
            return 'hand'
        if 'BATTLE' in s:
            return 'battle'
        if 'DECK' in s:
            return 'deck'
    return 'hand'


@dataclass
class GameCommand:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


class GameState:
    """Minimal GameState stub used by tests that only instantiate and query simple attributes."""
    def __init__(self, capacity: int = 0):
        self.capacity = capacity
        self.active_modifiers: List[Any] = []
        self.players: List[Dict[str, Any]] = []
        # basic game bookkeeping used by tests
        self.turn_number: int = 0
        self.current_phase: Optional[Phase] = None

    def add_modifier(self, m: Any):
        self.active_modifiers.append(m)

    # Minimal zone/player management for unit tests
    def _ensure_player(self, player_id: int):
        while len(self.players) <= player_id:
            ns = SimpleNamespace()
            ns.hand = []
            ns.mana_zone = []
            # provide both names used across tests
            ns.battle = []
            ns.battle_zone = ns.battle
            ns.deck = []
            ns.graveyard = []
            self.players.append(ns)

    def add_card_to_hand(self, player_id: int, card_id: int, instance_id: int):
        self._ensure_player(player_id)
        ci = SimpleNamespace()
        ci.card_id = card_id
        ci.instance_id = instance_id
        ci.is_tapped = False
        self.players[player_id].hand.append(ci)

    def add_test_card_to_battle(self, player_id: int, card_id: int, instance_id: int, is_tapped: bool, _):
        self._ensure_player(player_id)
        ci = SimpleNamespace()
        ci.card_id = card_id
        ci.instance_id = instance_id
        ci.is_tapped = is_tapped
        self.players[player_id].battle.append(ci)
        # ensure battle_zone alias updated
        self.players[player_id].battle_zone = self.players[player_id].battle

    def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int):
        self._ensure_player(player_id)
        ci = SimpleNamespace()
        ci.card_id = card_id
        ci.instance_id = instance_id
        self.players[player_id].deck.append(ci)

    def get_card_instance(self, instance_id: int):
        for p in self.players:
            for zone in ('hand', 'mana_zone', 'battle'):
                for ci in getattr(p, zone, []):
                    if getattr(ci, 'instance_id', None) == instance_id:
                        return ci
        return None

    # Simple command executor used by unit tests
    def execute_command(self, cmd: Any):
        # prefer cmd.execute if available
        if hasattr(cmd, 'execute'):
            cmd.execute(self)
            return

        # Fallback basic behavior
        if isinstance(cmd, TransitionCommand):
            # move from from_zone to to_zone
            player = self.players[cmd.player_id]
            src = _zone_name(cmd.from_zone)
            dst = _zone_name(cmd.to_zone)
            # find and remove using attribute access
            src_list = getattr(player, src, [])
            dst_list = getattr(player, dst, [])
            for i, ci in enumerate(list(src_list)):
                if getattr(ci, 'instance_id', None) == cmd.instance_id:
                    # remove from src_list and append to dst_list
                    src_list.pop(i)
                    dst_list.append(ci)
                    break
        elif isinstance(cmd, MutateCommand):
            ci = self.get_card_instance(cmd.instance_id)
            if ci is not None:
                if cmd.mutation_kind == MutationType.TAP:
                    ci.is_tapped = True
                elif cmd.mutation_kind == MutationType.POWER_MOD:
                    ci.power_mod = getattr(ci, 'power_mod', 0) + (cmd.amount or 0)
        elif isinstance(cmd, FlowCommand):
            # flow_type PHASE_CHANGE: payload is new phase int
            if getattr(cmd, 'flow_type', None) == FlowType.PHASE_CHANGE:
                self._prev_phase = getattr(self, 'current_phase', None)
                # accept either enum or int
                try:
                    self.current_phase = Phase(cmd.value) if isinstance(cmd.value, int) is False else Phase(cmd.value)
                except Exception:
                    self.current_phase = cmd.value
        elif isinstance(cmd, QueryCommand):
            self.pending_query = SimpleNamespace()
            self.pending_query.query_type = cmd.query_type
            self.pending_query.valid_targets = cmd.valid_targets
            self.waiting_for_user_input = True

    # ensure waiting flag present by default
    waiting_for_user_input: bool = False


@dataclass
class CardDefinition:
    id: int
    name: str = ""
    cost: int = 0


# --- Command subclasses (stubs) ---
class TransitionCommand(GameCommand):
    def __init__(self, instance_id, from_zone, to_zone, player_id):
        self.instance_id = instance_id
        self.from_zone = from_zone
        self.to_zone = to_zone
        self.player_id = player_id

    def execute(self, state: GameState):
        # perform move directly to avoid recursion
        player = state.players[self.player_id]
        src = _zone_name(self.from_zone)
        dst = _zone_name(self.to_zone)
        src_list = getattr(player, src, [])
        dst_list = getattr(player, dst, [])
        for i, ci in enumerate(list(src_list)):
            if getattr(ci, 'instance_id', None) == self.instance_id:
                src_list.pop(i)
                dst_list.append(ci)
                break

    def invert(self, state: GameState):
        inv = TransitionCommand(self.instance_id, self.to_zone, self.from_zone, self.player_id)
        state.execute_command(inv)


class MutateCommand(GameCommand):
    def __init__(self, instance_id, mutation_kind, amount: Optional[int] = None):
        self.instance_id = instance_id
        self.mutation_kind = mutation_kind
        self.amount = amount

    def execute(self, state: GameState):
        ci = state.get_card_instance(self.instance_id)
        if ci is None:
            return
        if self.mutation_kind == MutationType.TAP:
            ci.is_tapped = True
        elif self.mutation_kind == MutationType.POWER_MOD:
            ci.power_mod = getattr(ci, 'power_mod', 0) + (self.amount or 0)

    def invert(self, state: GameState):
        ci = state.get_card_instance(self.instance_id)
        if ci is None:
            return
        if self.mutation_kind == MutationType.TAP:
            ci.is_tapped = False
        elif self.mutation_kind == MutationType.POWER_MOD:
            ci.power_mod = getattr(ci, 'power_mod', 0) - (self.amount or 0)


class FlowCommand(GameCommand):
    def __init__(self, flow_type, value):
        self.flow_type = flow_type
        self.value = value

    def execute(self, state: GameState):
        if getattr(self, 'flow_type', None) == FlowType.PHASE_CHANGE:
            state._prev_phase = getattr(state, 'current_phase', None)
            try:
                state.current_phase = Phase(self.value) if isinstance(self.value, int) is False else Phase(self.value)
            except Exception:
                state.current_phase = self.value

    def invert(self, state: GameState):
        if hasattr(state, '_prev_phase'):
            state.current_phase = state._prev_phase


class QueryCommand(GameCommand):
    def __init__(self, query_type, valid_targets, constraints=None):
        self.query_type = query_type
        self.valid_targets = valid_targets
        self.constraints = constraints or {}

    def execute(self, state: GameState):
        state.pending_query = SimpleNamespace()
        state.pending_query.query_type = self.query_type
        state.pending_query.valid_targets = self.valid_targets
        state.waiting_for_user_input = True

    def invert(self, state: GameState):
        state.pending_query = None
        state.waiting_for_user_input = False


class DecideCommand(GameCommand):
    def __init__(self, choices: List[Any]):
        self.choices = choices


class EffectSystem:
    pass


class ActionDef:
    pass


class EffectActionType(Enum):
    TRANSITION = 'TRANSITION'
    GRANT_KEYWORD = 'GRANT_KEYWORD'
    MOVE_CARD = 'MOVE_CARD'
    FRIEND_BURST = 'FRIEND_BURST'
    APPLY_MODIFIER = 'APPLY_MODIFIER'
    DRAW_CARD = 'DRAW_CARD'
    ADD_MANA = 'ADD_MANA'
    DESTROY = 'DESTROY'
    RETURN_TO_HAND = 'RETURN_TO_HAND'
    SEND_TO_MANA = 'SEND_TO_MANA'
    TAP = 'TAP'
    UNTAP = 'UNTAP'
    MODIFY_POWER = 'MODIFY_POWER'
    BREAK_SHIELD = 'BREAK_SHIELD'
    LOOK_AND_ADD = 'LOOK_AND_ADD'
    SEARCH_DECK_BOTTOM = 'SEARCH_DECK_BOTTOM'
    MEKRAID = 'MEKRAID'
    REVOLUTION_CHANGE = 'REVOLUTION_CHANGE'
    COUNT_CARDS = 'COUNT_CARDS'
    GET_GAME_STAT = 'GET_GAME_STAT'
    REVEAL_CARDS = 'REVEAL_CARDS'
    RESET_INSTANCE = 'RESET_INSTANCE'
    REGISTER_DELAYED_EFFECT = 'REGISTER_DELAYED_EFFECT'
    SEARCH_DECK = 'SEARCH_DECK'
    SHUFFLE_DECK = 'SHUFFLE_DECK'
    ADD_SHIELD = 'ADD_SHIELD'
    SEND_SHIELD_TO_GRAVE = 'SEND_SHIELD_TO_GRAVE'
    SEND_TO_DECK_BOTTOM = 'SEND_TO_DECK_BOTTOM'
    MOVE_TO_UNDER_CARD = 'MOVE_TO_UNDER_CARD'
    CAST_SPELL = 'CAST_SPELL'
    PUT_CREATURE = 'PUT_CREATURE'
    COST_REFERENCE = 'COST_REFERENCE'
    SELECT_NUMBER = 'SELECT_NUMBER'
    SUMMON_TOKEN = 'SUMMON_TOKEN'
    DISCARD = 'DISCARD'
    PLAY_FROM_ZONE = 'PLAY_FROM_ZONE'
    LOOK_TO_BUFFER = 'LOOK_TO_BUFFER'
    SELECT_FROM_BUFFER = 'SELECT_FROM_BUFFER'
    PLAY_FROM_BUFFER = 'PLAY_FROM_BUFFER'
    MOVE_BUFFER_TO_ZONE = 'MOVE_BUFFER_TO_ZONE'
    SELECT_OPTION = 'SELECT_OPTION'
    RESOLVE_BATTLE = 'RESOLVE_BATTLE'


class InstructionOp(Enum):
    NOP = 'NOP'


class MutationType(Enum):
    GAME_STAT = 'GAME_STAT'
    TAP = 'TAP'
    POWER_MOD = 'POWER_MOD'


class FlowType(Enum):
    SEQUENTIAL = 'SEQUENTIAL'
    PHASE_CHANGE = 'PHASE_CHANGE'


def register_card_data(*args, **kwargs):
    return None


class PhaseManager:
    pass


class GenericCardSystem:
    @staticmethod
    def resolve_action(state: GameState, action_def, player_id: int):
        # Minimal resolver: implement DRAW_CARD behavior used by tests
        if getattr(action_def, 'type', None) == EffectActionType.DRAW_CARD:
            count = int(getattr(action_def, 'value1', 1) or 1)
            # draw from top of deck (treat deck end as top)
            for _ in range(count):
                # find a player id: if player_id == -1, default to 0
                pid = player_id if player_id >= 0 else 0
                if pid >= len(state.players):
                    continue
                deck = getattr(state.players[pid], 'deck', [])
                if not deck:
                    continue
                ci = deck.pop()  # take top
                state.players[pid].hand.append(ci)
        else:
            # No-op for other action types in this minimal stub
            return None


class ConditionDef:
    pass


class FilterDef:
    pass


class TargetScope(Enum):
    PLAYER_SELF = 'PLAYER_SELF'


# Basic TriggerType and EffectDef used by some integration tests
class TriggerType(Enum):
    ON_PLAY = 'ON_PLAY'
    ON_SUMMON = 'ON_SUMMON'
    ON_DESTROY = 'ON_DESTROY'


@dataclass
class EffectDef:
    type: EffectActionType = EffectActionType.TRANSITION
    value1: Optional[int] = None
    value2: Optional[int] = None
    str_val: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None



class CardKeywords:
    pass


@dataclass
class CardData:
    id: int
    name: str
    cost: int
    civilization: str
    power: int
    type: str
    keywords: List[Any]
    effects: List[Any]


class CardRegistry:
    pass


# --- Minimal Tensor converter used by unit tests ---
class TensorConverter:
    INPUT_SIZE = 16

    @staticmethod
    def convert_to_tensor(game_state: GameState, player_id: int, card_db: Dict[int, Any]):
        # Produce a deterministic fixed-size vector representing minimal state for tests
        vec = [0.0] * TensorConverter.INPUT_SIZE
        # simple encoding: number of cards in hand/mana/battle for player
        try:
            p = game_state.players[player_id]
            vec[0] = float(len(getattr(p, 'hand', [])))
            vec[1] = float(len(getattr(p, 'mana_zone', [])))
            vec[2] = float(len(getattr(p, 'battle', [])))
        except Exception:
            pass
        return vec

