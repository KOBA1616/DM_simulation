# Fallback shim for dm_ai_module to aid tests in pure-Python environments
# This module tries to import the compiled extension first; if missing or
# missing symbols, it defines lightweight placeholders so tests can import.
from enum import Enum

try:
    import dm_ai_module as _native  # type: ignore
except Exception:
    _native = None

# Helper to copy attribute from native if present
def _get_attr(name, default):
    if _native is not None and hasattr(_native, name):
        return getattr(_native, name)
    return default

# Minimal Enums / placeholders
class Civilization(Enum):
    FIRE = 1
    WATER = 2
    NATURE = 3
    LIGHT = 4
    DARKNESS = 5

class CardType(Enum):
    CREATURE = 1
    SPELL = 2

class CardKeywords(int):
    pass

class PassiveType(Enum):
    NONE = 0

class PassiveEffect:
    def __init__(self, *args, **kwargs):
        pass

class FilterDef(dict):
    pass

class PlayerStub:
    def __init__(self):
        self.hand = []
        self.deck = []
        self.battle_zone = []
        self.graveyard = []
        self.mana_zone = []
        self.shield_zone = []

class GameState:
    def __init__(self, *args, **kwargs):
        # minimal placeholder for tests; real implementation lives in native module
        self.game_over = False
        self.turn_number = 0
        self.players = [PlayerStub(), PlayerStub()]
        self.active_player_id = 0

    def setup_test_duel(self):
        # no-op placeholder
        return

    def add_test_card_to_battle(self, player_id, card_id, instance_id, tapped, sick):
        p = self.players[player_id]
        # minimal card stub
        class CardStub:
             def __init__(self, iid, cid):
                 self.instance_id = iid
                 self.card_id = cid
                 self.is_tapped = tapped
                 self.summoning_sickness = sick
                 self.id = iid
        c = CardStub(instance_id, card_id)
        p.battle_zone.append(c)

    def add_card_to_deck(self, player_id, card_id, instance_id):
        p = self.players[player_id]
        class CardStub:
             def __init__(self, iid, cid):
                 self.instance_id = iid
                 self.card_id = cid
        c = CardStub(instance_id, card_id)
        p.deck.append(c)

class CardDefinition:
    def __init__(self):
        pass

# Export names, preferring native implementations when available
GameState = _get_attr('GameState', GameState)
CardDefinition = _get_attr('CardDefinition', CardDefinition)
Civilization = _get_attr('Civilization', Civilization)
CardType = _get_attr('CardType', CardType)
CardKeywords = _get_attr('CardKeywords', CardKeywords)
PassiveEffect = _get_attr('PassiveEffect', PassiveEffect)
PassiveType = _get_attr('PassiveType', PassiveType)
FilterDef = _get_attr('FilterDef', FilterDef)

# --- Minimal shim implementations for APIs used by tests when native extension missing ---
_batch_callback = None

def set_batch_callback(cb):
    global _batch_callback
    _batch_callback = cb

def has_batch_callback():
    return _batch_callback is not None

def clear_batch_callback():
    global _batch_callback
    _batch_callback = None

class ActionEncoder:
    # reasonable default for tests
    TOTAL_ACTION_SIZE = 10

class NeuralEvaluator:
    def __init__(self, card_db):
        self.card_db = card_db

    def evaluate(self, batch):
        # call registered batch callback if present, otherwise return defaults
        if _batch_callback is not None:
            return _batch_callback(batch)
        policies = [[0.0] * ActionEncoder.TOTAL_ACTION_SIZE for _ in batch]
        values = [0.0 for _ in batch]
        return policies, values

# Prefer native implementations when available
ActionEncoder = _get_attr('ActionEncoder', ActionEncoder)
NeuralEvaluator = _get_attr('NeuralEvaluator', NeuralEvaluator)
set_batch_callback = _get_attr('set_batch_callback', set_batch_callback)
has_batch_callback = _get_attr('has_batch_callback', has_batch_callback)
clear_batch_callback = _get_attr('clear_batch_callback', clear_batch_callback)

class CommandDef:
    def __init__(self, *args, **kwargs):
        pass

class CardData:
    def __init__(self, *args, **kwargs):
        pass

class CommandSystem:
    @staticmethod
    def execute_command(state, cmd, *args, **kwargs):
        pass

class EffectResolver:
    @staticmethod
    def resolve_action(state, action, card_db):
        pass

class PhaseManager:
    @staticmethod
    def next_phase(state, card_db):
        pass

def register_card_data(data):
    pass

class CommandType(Enum):
    TRANSITION = 1

class TargetScope(Enum):
    PLAYER_SELF = 1
    SELF = 1

class Zone(Enum):
    DECK = 1
    HAND = 2
    GRAVEYARD = 3
    MANA = 4
    BATTLE_ZONE = 5
    SHIELD_ZONE = 6

# Also export CommandType/Zone if present in native module to satisfy other code
CommandType = _get_attr('CommandType', CommandType)
Zone = _get_attr('Zone', Zone)
CommandDef = _get_attr('CommandDef', CommandDef)
TargetScope = _get_attr('TargetScope', TargetScope)
CardData = _get_attr('CardData', CardData)
CommandSystem = _get_attr('CommandSystem', CommandSystem)
register_card_data = _get_attr('register_card_data', register_card_data)
EffectResolver = _get_attr('EffectResolver', EffectResolver)
PhaseManager = _get_attr('PhaseManager', PhaseManager)
