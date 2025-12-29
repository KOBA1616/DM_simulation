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
    def __init__(self):
        pass

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

# Also export CommandType/Zone if present in native module to satisfy other code
CommandType = _get_attr('CommandType', None)
Zone = _get_attr('Zone', None)
