# -*- coding: utf-8 -*-
import sys
import os

# Try to import dm_ai_module
try:
    # If strictly needed, we could append bin/ to path here, but usually app handles it.
    import dm_ai_module
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

def _get_enum_names(enum_cls):
    """Helper to extract names from pybind11 enum."""
    # pybind11 enums usually have __members__ dict
    if hasattr(enum_cls, '__members__'):
        return list(enum_cls.__members__.keys())
    # Fallback if __members__ missing (older pybind11?) but usually it's there.
    # We can also usedir(enum_cls) but that includes internal methods.
    return [name for name in dir(enum_cls) if not name.startswith('_') and name not in ['name', 'value']]

# =============================================================================
# Card Types
# =============================================================================
if _HAS_MODULE and hasattr(dm_ai_module, 'CardType'):
    CARD_TYPES = _get_enum_names(dm_ai_module.CardType)
else:
    # Fallback to hardcoded list matching C++ CardType
    CARD_TYPES = [
        "CREATURE",
        "SPELL",
        "EVOLUTION_CREATURE",
        "CROSS_GEAR",
        "PSYCHIC_CREATURE",
        "GR_CREATURE",
        "TAMASEED",
        "CASTLE"
    ]

# =============================================================================
# Zones
# =============================================================================
# Note: The Engine explicitly expects these string values in JSON (FilterDef, etc.)
# which differ from the C++ Enum names (e.g. "BATTLE_ZONE" vs "BATTLE").
# Do not change these strings unless the C++ JsonLoader/TargetUtils are updated.
ZONES = [
    "BATTLE_ZONE",
    "MANA_ZONE",
    "HAND",
    "GRAVEYARD",
    "SHIELD_ZONE",
    "DECK"
]

# Extended Zone options for UI (Destinations, etc.)
ZONES_EXTENDED = ZONES + ["DECK_BOTTOM", "DECK_TOP", "NONE"]

# =============================================================================
# Civilizations
# =============================================================================
if _HAS_MODULE and hasattr(dm_ai_module, 'Civilization'):
    CIVILIZATIONS = [c for c in _get_enum_names(dm_ai_module.Civilization) if c != 'NONE']
else:
    CIVILIZATIONS = ["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"]

# =============================================================================
# Command Types (GameCommand)
# =============================================================================
if _HAS_MODULE and hasattr(dm_ai_module, 'CommandType'):
    COMMAND_TYPES = _get_enum_names(dm_ai_module.CommandType)
else:
    COMMAND_TYPES = [
        "TRANSITION", "MUTATE", "FLOW", "QUERY",
        "DRAW_CARD", "DISCARD", "DESTROY", "MANA_CHARGE",
        "TAP", "UNTAP", "POWER_MOD", "ADD_KEYWORD",
        "RETURN_TO_HAND", "BREAK_SHIELD", "SEARCH_DECK", "SHIELD_TRIGGER",
        "NONE"
    ]

# =============================================================================
# Action Types (EffectActionType)
# =============================================================================
if _HAS_MODULE and hasattr(dm_ai_module, 'EffectActionType'):
    ACTION_TYPES = _get_enum_names(dm_ai_module.EffectActionType)
else:
    ACTION_TYPES = [
        "DRAW_CARD", "ADD_MANA", "DESTROY", "RETURN_TO_HAND", "SEND_TO_MANA",
        "TAP", "UNTAP", "MODIFY_POWER", "BREAK_SHIELD", "LOOK_AND_ADD",
        "SUMMON_TOKEN", "SEARCH_DECK_BOTTOM", "MEKRAID", "DISCARD",
        "PLAY_FROM_ZONE", "COST_REFERENCE", "REVOLUTION_CHANGE", "COUNT_CARDS",
        "GET_GAME_STAT", "APPLY_MODIFIER", "REVEAL_CARDS", "REGISTER_DELAYED_EFFECT",
        "RESET_INSTANCE", "SEARCH_DECK", "SHUFFLE_DECK", "ADD_SHIELD",
        "SEND_SHIELD_TO_GRAVE", "SEND_TO_DECK_BOTTOM", "MOVE_TO_UNDER_CARD",
        "SELECT_NUMBER", "FRIEND_BURST", "GRANT_KEYWORD", "MOVE_CARD",
        "CAST_SPELL", "PUT_CREATURE", "SELECT_OPTION", "RESOLVE_BATTLE", "NONE"
    ]

# =============================================================================
# UI Specific Lists (Editor Subsets)
# =============================================================================
# List of Actions supported/exposed in the Editor UI
EDITOR_ACTION_TYPES = [
    "MOVE_CARD",
    "DRAW_CARD",
    "TAP",
    "UNTAP",
    "SEARCH_DECK",
    "MEKRAID",
    "COST_REFERENCE",
    "COST_REDUCTION",
    "NONE",
    "BREAK_SHIELD",
    "LOOK_AND_ADD",
    "SUMMON_TOKEN",
    "PLAY_FROM_ZONE",
    "REVOLUTION_CHANGE",
    "MEASURE_COUNT",
    "APPLY_MODIFIER",
    "REVEAL_CARDS",
    "REGISTER_DELAYED_EFFECT",
    "RESET_INSTANCE",
    "FRIEND_BURST",
    "GRANT_KEYWORD",
    "SELECT_OPTION"
]

# Actions that are deprecated and should be replaced (e.g. by MOVE_CARD)
LEGACY_ACTION_TYPES = [
    "DESTROY", "RETURN_TO_HAND", "ADD_MANA", "DISCARD",
    "SEND_TO_DECK_BOTTOM", "SEND_SHIELD_TO_GRAVE", "SEND_TO_MANA"
]

# =============================================================================
# Grantable Keywords
# =============================================================================
# Keywords that can be granted via ADD_KEYWORD / GRANT_KEYWORD
GRANTABLE_KEYWORDS = [
    "speed_attacker",
    "blocker",
    "slayer",
    "double_breaker",
    "triple_breaker",
    "world_breaker",
    "mach_fighter",
    "g_strike",
    "unblockable",
    "shield_burn",
    "just_diver",
    "shield_trigger",
    "power_attacker",
    "CANNOT_ATTACK",
    "CANNOT_BLOCK",
    "CANNOT_ATTACK_OR_BLOCK"
]
