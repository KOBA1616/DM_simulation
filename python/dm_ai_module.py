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

class GameState:
    def __init__(self, *args, **kwargs):
        # minimal placeholder for tests; real implementation lives in native module
        self.game_over = False
        self.turn_number = 0

    def setup_test_duel(self):
        # no-op placeholder
        return

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

# Also export CommandType/Zone if present in native module to satisfy other code
CommandType = _get_attr('CommandType', None)
Zone = _get_attr('Zone', None)
