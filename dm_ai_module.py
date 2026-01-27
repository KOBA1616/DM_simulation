"""Minimal Python fallback shim for dm_ai_module used by tests and tools.

This simplified shim implements a small subset of the native API used by
the test-suite and tooling. It's intentionally compact and defensive so
it runs reliably when the native extension is unavailable.
"""

from enum import IntEnum
import warnings
from typing import Any, Dict, List, Optional
import json
import os

# Flag indicating native extension not present
IS_NATIVE = False


class ActionType(IntEnum):
    NONE = 0
    PASS = 1
    MANA_CHARGE = 2
    PLAY_CARD = 3
    RESOLVE_EFFECT = 8
    ATTACK_PLAYER = 5
    ATTACK_CREATURE = 6


class CommandType(IntEnum):
    NONE = 0
    PLAY_FROM_ZONE = 1
    MANA_CHARGE = 2
    ATTACK = 3
    PASS = 4


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
        self.shield_zone: List[CardStub] = []
        self.graveyard: List[CardStub] = []
        self.deck: List[int] = []
        self.life: int = 20


class GameState:
    def __init__(self):
        self.players: List[Player] = [Player(0), Player(1)]
        self.current_phase = 2
        self.active_player_id = 0
        self.pending_effects: List[Any] = []
        self.turn_number = 1
        self.game_over = False
        self.winner = -1
        class _Exec:
            pass
        self.execution_context = _Exec()
        self.execution_context.variables = {}

    def get_card_instance(self, instance_id: int) -> Optional[CardStub]:
        for p in self.players:
            for c in p.hand + p.battle_zone + p.mana_zone + p.graveyard:
                try:
                    if getattr(c, 'instance_id', None) == instance_id:
                        return c
                except Exception:
                    continue
        return None

    def add_card_to_hand(self, player: int, card_id: int, instance_id: Optional[int] = None, count: int = 1):
        if instance_id is not None and count == 1:
            c = CardStub(card_id, instance_id)
            self.players[player].hand.append(c)
            return c
        for _ in range(count):
            c = CardStub(card_id)
            self.players[player].hand.append(c)
        return self.players[player].hand[-1] if self.players[player].hand else None

    def add_test_card_to_battle(self, player: int, card_id: int, instance_id: Optional[int] = None, tapped: bool = False, sick: bool = False):
        c = CardStub(card_id, instance_id)
        c.is_tapped = tapped
        c.sick = sick
        self.players[player].battle_zone.append(c)
        return c

    def add_card_to_mana(self, player: int, card_id: int, instance_id: Optional[int] = None, count: int = 1):
        if instance_id is not None and count == 1:
            c = CardStub(card_id, instance_id)
            self.players[player].mana_zone.append(c)
            return c
        for _ in range(count):
            c = CardStub(card_id)
            self.players[player].mana_zone.append(c)
        return self.players[player].mana_zone[-1] if self.players[player].mana_zone else None

    def set_deck(self, player: int, deck_ids: List[int]):
        try:
            self.players[player].deck = list(deck_ids)
        except Exception:
            pass

    def get_next_instance_id(self) -> int:
        CardStub._iid += 1
        return CardStub._iid

    def setup_test_duel(self):
        # Minimal reset for tests
        self.players = [Player(0), Player(1)]
        self.current_phase = 2
        self.active_player_id = 0
        self.pending_effects = []
        self.turn_number = 1
        self.game_over = False


class Action:
    """Minimal legacy Action object for compatibility.

    Fields: `type`, `source_instance_id`, `target_player` are used by compat layer/tests.
    """
    def __init__(self, type: Any = ActionType.NONE, source_instance_id: Optional[int] = None, target_player: int = 255):
        warnings.warn(
            "`Action` is deprecated — migrate to Command-based APIs (`Command`/`generate_legal_commands`) and unified command dicts.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.type = type
        self.source_instance_id = source_instance_id
        self.target_player = target_player

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': getattr(self.type, 'name', self.type),
            'source_instance_id': self.source_instance_id,
            'target_player': self.target_player
        }


# Backwards-compatible aliases/legacy names expected by tests
# GameCommand alias will be defined after CommandDef is declared


class GameResult(IntEnum):
    NONE = 0
    P1_WIN = 1
    P2_WIN = 2
    DRAW = 3


class CardType(IntEnum):
    UNKNOWN = 0
    CREATURE = 1
    SPELL = 2
    # Tests primarily check presence; extend as needed later.
 


class GameInstance:
    def __init__(self, seed: int = 0, card_db: Any = None):
        self.state = GameState()
        self.card_db = card_db

    def start_game(self):
        self.state.current_phase = 2
        self.state.active_player_id = 0

    def execute_action(self, action: Any):
        # Minimal semantics: move by instance_id for MANA_CHARGE or PLAY_CARD
        if getattr(action, 'type', None) in (ActionType.MANA_CHARGE, ActionType.PLAY_CARD):
            pid = getattr(self.state, 'active_player_id', 0)
            iid = getattr(action, 'source_instance_id', getattr(action, 'instance_id', None))
            if iid is not None:
                inst = self.state.get_card_instance(iid)
                if inst:
                    # remove from any zone and append to target zone
                    for zone in ('hand', 'battle_zone', 'mana_zone'):
                        z = getattr(self.state.players[pid], zone, [])
                        for i, c in enumerate(list(z)):
                            if getattr(c, 'instance_id', None) == iid:
                                try:
                                    z.pop(i)
                                except Exception:
                                    pass
                                break
                    if action.type == ActionType.MANA_CHARGE:
                        self.state.players[pid].mana_zone.append(inst)
                    else:
                        self.state.players[pid].battle_zone.append(inst)


class CommandDef:
    def __init__(self):
        self.type = None
        self.instance_id = 0
        self.target_instance = 0
        self.owner_id = 0
        self.player_id = 0
        self.from_zone = ''
        self.to_zone = ''
        self.mutation_kind = ''
        self.input_value_key = ''
        self.output_value_key = ''
        self.target_filter = None
        self.target_group = None
        self.amount = 0
        self.str_param = ''
        self.optional = False

# Backwards-compatible alias defined after CommandDef
GameCommand = CommandDef  # legacy alias


class FlowType(IntEnum):
    SET_ATTACK_SOURCE = 1
    SET_ATTACK_TARGET = 2
    DECLARE_ATTACK = 3


class FlowCommand:
    def __init__(self, flow_type: Any, arg: Any = None):
        self.type = flow_type
        self.arg = arg
        # Some tests expect `new_value` attribute for flow commands
        self.new_value = arg


def get_action_type(obj: Any) -> Any:
    """Return the 'type' of an action-like object in a normalized form.

    Supports legacy `Action` objects, command dicts, and enum/int/string types.
    """
    if obj is None:
        return None
    try:
        if isinstance(obj, dict):
            return obj.get('type')
    except Exception:
        pass
    return getattr(obj, 'type', None)


def is_action_type(obj: Any, expected: Any) -> bool:
    """Safe comparison for action types.

    Allows `obj` to be an `Action` object, a command `dict`, an `IntEnum`,
    or plain ints/strings. `expected` is typically an `ActionType` member.
    """
    val = get_action_type(obj)
    # Direct enum comparison
    try:
        if isinstance(val, IntEnum):
            return val == expected
    except Exception:
        pass
    # String names in command dicts (e.g. "PASS")
    if isinstance(val, str) and isinstance(expected, IntEnum):
        return val == expected.name or val == str(expected.value)
    # Numeric comparisons
    if isinstance(val, int) and isinstance(expected, IntEnum):
        return val == expected.value
    try:
        return val == expected
    except Exception:
        return False


class FilterDef:
    def __init__(self):
        self.zones: List[str] = []
        self.types: List[Any] = []
        self.owner: Optional[int] = None
        self.count: int = 0


class TargetScope(IntEnum):
    NONE = 0
    ALL = 1
    OPPONENT = 2
    SELF = 3


class CommandSystem:
    @staticmethod
    def execute_command(state: GameState, cmd: Any, source_id: int = -1, player_id: int = 0, ctx: Any = None) -> None:
        # Support dict commands and simple CommandDef/object shapes
        try:
            # normalize
            if isinstance(cmd, dict):
                cdict = cmd
                t = cdict.get('type')
                # try name
                tname = t if isinstance(t, str) else (getattr(t, 'name', None) if t is not None else None)
                iid = None
                for key in ('instance_id', 'source_instance_id', 'source_id', 'source'):
                    if key in cdict:
                        try:
                            iid = int(cdict[key])
                        except Exception:
                            iid = None
                # player id
                try:
                    player_id = int(cdict.get('player_id', player_id))
                except Exception:
                    pass
                if tname == 'MANA_CHARGE':
                    pid = getattr(state, 'active_player_id', player_id)
                    if iid is not None:
                        for i, c in enumerate(list(state.players[pid].hand)):
                            if getattr(c, 'instance_id', None) == iid:
                                state.players[pid].hand.pop(i)
                                state.players[pid].mana_zone.append(c)
                                return
                    # fallback: add stub
                    cid = cdict.get('card_id', 0)
                    state.players[pid].mana_zone.append(CardStub(int(cid) if cid is not None else 0))
                    return
            else:
                # object-like
                t = getattr(cmd, 'type', None)
                tname = getattr(t, 'name', t) if t is not None else None
                iid = getattr(cmd, 'instance_id', getattr(cmd, 'source_instance_id', None))
                pid = getattr(cmd, 'player_id', getattr(cmd, 'owner_id', player_id))
                if tname == 'MANA_CHARGE' and iid is not None:
                    for i, c in enumerate(list(state.players[pid].hand)):
                        if getattr(c, 'instance_id', None) == iid:
                            state.players[pid].hand.pop(i)
                            state.players[pid].mana_zone.append(c)
                            return
        except Exception:
            pass

        # Minimal implementation for SELECT_TARGET-like commands used in tests
        try:
            # Normalize input
            cdict = cmd if isinstance(cmd, dict) else None
            if cdict:
                typ = str(cdict.get('type', '')).upper()
            else:
                t = getattr(cmd, 'type', None)
                typ = getattr(t, 'name', str(t) if t is not None else '')

            # SELECT_TARGET: populate state.execution_context.variables[out_key]
            if typ == 'SELECT_TARGET':
                target_group = cdict.get('target_group', '') if cdict else getattr(cmd, 'target_group', '')
                amount = int(cdict.get('amount', 1)) if cdict else int(getattr(cmd, 'amount', 1))
                out_key = cdict.get('output_value_key', 'selected') if cdict else getattr(cmd, 'output_value_key', 'selected')

                candidates = []
                if target_group in ('PLAYER_SELF', 'PLAYER_SELF'):
                    candidates = list(state.players[player_id].battle_zone)
                elif target_group in ('PLAYER_OPPONENT', 'PLAYER_OPP'):
                    candidates = list(state.players[1 - player_id].battle_zone)

                selected = [getattr(c, 'instance_id', None) for c in candidates[:amount]]

                if not hasattr(state, 'execution_context') or getattr(state, 'execution_context', None) is None:
                    class _Ctx: pass
                    state.execution_context = _Ctx()
                    state.execution_context.variables = {}

                state.execution_context.variables[out_key] = selected
                return

            # DESTROY: move instances listed in execution_context[input_value_key] to graveyard
            if typ == 'DESTROY':
                in_key = cdict.get('input_value_key') if cdict else getattr(cmd, 'input_value_key', None)
                if in_key and hasattr(state, 'execution_context') and hasattr(state.execution_context, 'variables'):
                    vals = state.execution_context.variables.get(in_key, []) or []
                    for iid in list(vals):
                        for p in state.players:
                            for zone_name in ('battle_zone', 'mana_zone', 'hand'):
                                zone = getattr(p, zone_name, [])
                                for i, c in enumerate(list(zone)):
                                    if getattr(c, 'instance_id', None) == iid:
                                        try:
                                            zone.pop(i)
                                        except Exception:
                                            pass
                                        p.graveyard.append(c)
                                        break
                    return
        except Exception:
            pass


class JsonLoader:
    @staticmethod
    def load_cards(filepath: str) -> Dict[int, Any]:
        final = filepath
        if not os.path.exists(final):
            alt = os.path.join(os.path.dirname(__file__), filepath)
            if os.path.exists(alt):
                final = alt
        try:
            with open(final, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                out: Dict[int, Any] = {}
                for item in data:
                    try:
                        out[int(item.get('id'))] = item
                    except Exception:
                        continue
                return out
            if isinstance(data, dict):
                out: Dict[int, Any] = {}
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


class ParallelRunner:
    def __init__(self, card_db: Any, sims: int, batch_size: int):
        self.card_db = card_db
        self.sims = sims
        self.batch_size = batch_size

    def play_games(self, initial_states: List[Any], evaluator_func: Any, temperature: float, add_noise: bool, threads: int) -> List[Any]:
        # Return a result object per initial state so tests can validate structure.
        results = []
        for _ in initial_states:
            class _R: pass
            r = _R()
            r.result = None
            r.winner = None
            results.append(r)
        return results


def create_parallel_runner(card_db: Any, sims: int, batch_size: int) -> ParallelRunner:
    return ParallelRunner(card_db, sims, batch_size)


def index_to_command(action_index: int, state: Any, card_db: Any = None) -> Dict[str, Any]:
    # Minimal mapping: treat small indices as mana/play
    if action_index == 0:
        return {'type': 'PASS'}
    if action_index < 20:
        return {'type': 'MANA_CHARGE', 'slot_index': action_index}
    return {'type': 'PLAY_FROM_ZONE', 'slot_index': action_index - 20}


def generate_commands(state: Any, card_db: Any = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pid = getattr(state, 'active_player_id', 0)
    p = state.players[pid]
    out.append({'type': 'PASS'})
    for i, c in enumerate(list(p.hand)):
        out.append({'type': 'MANA_CHARGE', 'instance_id': getattr(c, 'instance_id', None)})
        out.append({'type': 'PLAY_FROM_ZONE', 'instance_id': getattr(c, 'instance_id', None)})
    return out


class DataCollector:
    def collect_data_batch_heuristic(self, count: int, include_history: bool, some_flag: bool):
        class Batch:
            def __init__(self):
                self.values = []
        return Batch()


class PhaseManager:
    @staticmethod
    def next_phase(state: GameState, card_db: Any) -> None:
        """Advance the phase on the Python stub.

        Uses the same integer heuristics as EngineCompat fallback: 2->3->4->5->2.
        """
        try:
            cur = getattr(state, 'current_phase', None)
            try:
                cur_val = int(cur)
            except Exception:
                # If it's an enum-like, try to get .value
                try:
                    cur_val = int(getattr(cur, 'value', 0))
                except Exception:
                    cur_val = 0

            if cur_val >= 2 and cur_val < 5:
                nxt = cur_val + 1
            elif cur_val == 5:
                nxt = 2
            else:
                nxt = cur_val + 1

            try:
                setattr(state, 'current_phase', nxt)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def start_game(state: GameState, card_db: Any) -> None:
        try:
            setattr(state, 'current_phase', 2)
            setattr(state, 'active_player_id', 0)
        except Exception:
            pass

    @staticmethod
    def check_game_over(state: GameState) -> bool:
        try:
            return bool(getattr(state, 'game_over', False))
        except Exception:
            return False


__all__ = [
    'IS_NATIVE', 'ActionType', 'CommandType', 'CardStub', 'Player', 'GameState', 'GameInstance',
    'CommandDef', 'FilterDef', 'TargetScope', 'CommandSystem', 'JsonLoader', 'ParallelRunner',
    'create_parallel_runner', 'index_to_command', 'generate_commands', 'generate_legal_commands', 'Command', 'CommandGenerator', 'PhaseManager',
    'Action', 'GameCommand', 'GameResult', 'CardType', 'FlowType', 'FlowCommand', 'DataCollector'
]

