# -*- coding: utf-8 -*-
import sys
import os

"""
Central Constants Definition.

Policy:
This file serves as the Single Source of Truth for Python-side constants,
enumerations, and configuration lists that mirror C++ engine values or
define UI behavior.

1.  **Mirroring C++**: Where possible, values are dynamically loaded from `dm_ai_module`.
2.  **UI Definitions**: Lists like `EDITOR_ACTION_TYPES` define the available options in the GUI.
3.  **Translations**: Translations should be handled by `dm_toolkit.gui.localization`, not here.
"""

# Try to import dm_ai_module
try:
    # If strictly needed, we could append bin/ to path here, but usually app handles it.
    import dm_ai_module
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

from typing import Any, List


def _get_enum_names(enum_cls: Any) -> List[str]:
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

# Append pseudo-types for Filters
if "ELEMENT" not in CARD_TYPES:
    CARD_TYPES.append("ELEMENT")

# =============================================================================
# Target Scope (Unified)
# =============================================================================
# Unified scope constants for both trigger effects and static abilities.
# Replaces the separate PLAYER_SELF/PLAYER_OPPONENT (triggers) and SELF/OPPONENT (statics).
class TargetScope:
    """
    Unified target scope constants.
    
    Usage:
    - Static abilities: Use SELF/OPPONENT/ALL for 'scope' field
    - Trigger effects: Use SELF/OPPONENT for 'target_group' field
    - Filters: Use SELF/OPPONENT for 'owner' field
    """
    SELF = "SELF"
    OPPONENT = "OPPONENT"
    ALL = "ALL"
    
    # Legacy aliases for backward compatibility
    PLAYER_SELF = "SELF"
    PLAYER_OPPONENT = "OPPONENT"
    
    @classmethod
    def normalize(cls, value: str) -> str:
        """Normalize legacy PLAYER_* values to unified format."""
        if value == "PLAYER_SELF":
            return cls.SELF
        elif value == "PLAYER_OPPONENT":
            return cls.OPPONENT
        return value
    
    @classmethod
    def all_values(cls) -> list:
        """Get all valid scope values."""
        return [cls.SELF, cls.OPPONENT, cls.ALL]

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

# Command System Normalized Zones (for from_zone / to_zone in Commands)
# These align with C++ Zone Enum and dm_ai_module.Zone
COMMAND_ZONES = [
    "BATTLE",
    "MANA",
    "HAND",
    "GRAVEYARD",
    "SHIELD",
    "DECK"
]

# Extended Zone options for UI (Destinations, etc.)
ZONES_EXTENDED = COMMAND_ZONES + ["DECK_BOTTOM", "DECK_TOP", "NONE"]

# Legacy Map for compatibility
LEGACY_ZONE_MAP = {
    "BATTLE_ZONE": "BATTLE",
    "MANA_ZONE": "MANA",
    "SHIELD_ZONE": "SHIELD"
}

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
    # Fallback list should mirror docs/api/command_spec.md (best-effort)
    COMMAND_TYPES = [
        "TRANSITION",
        "REPLACE_CARD_MOVE",
        "MUTATE",
        "FLOW",
        "QUERY",
        "DRAW_CARD",
        "DISCARD",
        "DESTROY",
        "MANA_CHARGE",
        "TAP",
        "UNTAP",
        "POWER_MOD",
        "ADD_KEYWORD",
        "RETURN_TO_HAND",
        "BREAK_SHIELD",
        "SEARCH_DECK",
        "SHIELD_TRIGGER",
        "ATTACK_PLAYER",
        "ATTACK_CREATURE",
        "BLOCK",
        "RESOLVE_BATTLE",
        "RESOLVE_PLAY",
        "RESOLVE_EFFECT",
        "SHUFFLE_DECK",
        "LOOK_AND_ADD",
        "MEKRAID",
        "REVEAL_CARDS",
        "PLAY_FROM_ZONE",
        "CAST_SPELL",
        "SUMMON_TOKEN",
        "SHIELD_BURN",
        "SELECT_NUMBER",
        "CHOICE",
        "LOOK_TO_BUFFER",
        "REVEAL_TO_BUFFER",
        "SELECT_FROM_BUFFER",
        "PLAY_FROM_BUFFER",
        "MOVE_BUFFER_TO_ZONE",
        "FRIEND_BURST",
        "REGISTER_DELAYED_EFFECT",
        "NONE",
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
    "ADD_KEYWORD",
    "ADD_SHIELD",
    "APPLY_MODIFIER",
    "BREAK_SHIELD",
    "CANNOT_PUT_CREATURE",
    "CANNOT_SUMMON_CREATURE",
    "CAST_SPELL",
    "CHOICE",
    "COST_REDUCTION",
    "COST_REFERENCE",
    "DECLARE_NUMBER",
    "DESTROY",
    "DISCARD",
    "DRAW_CARD",
    "ELSE",
    "FLOW",
    "FRIEND_BURST",
    "GAME_RESULT",
    "GRANT_KEYWORD",
    "IF",
    "IF_ELSE",
    "LOOK_AND_ADD",
    "LOOK_TO_BUFFER",
    "MANA_CHARGE",
    "MEASURE_COUNT",
    "MEKRAID",
    "MOVE_BUFFER_TO_ZONE",
    "MOVE_CARD",
    "MOVE_TO_UNDER_CARD",
    "MUTATE",
    "NONE",
    "PLAYER_CANNOT_ATTACK",
    "PLAY_FROM_BUFFER",
    "PLAY_FROM_ZONE",
    "POWER_MOD",
    "PUT_CREATURE",
    "QUERY",
    "REGISTER_DELAYED_EFFECT",
    "REPLACE_CARD_MOVE",
    "RESET_INSTANCE",
    "RESOLVE_BATTLE",
    "RETURN_TO_HAND",
    "REVEAL_CARDS",
    "REVEAL_TO_BUFFER",
    "REVOLUTION_CHANGE",
    "SEARCH_DECK",
    "SEARCH_DECK_BOTTOM",
    "SELECT_FROM_BUFFER",
    "SELECT_NUMBER",
    "SELECT_OPTION",
    "SEND_SHIELD_TO_GRAVE",
    "SEND_TO_DECK_BOTTOM",
    "SHIELD_BURN",
    "SHIELD_TRIGGER",
    "SHUFFLE_DECK",
    "SPELL_RESTRICTION",
    "STAT",
    "SUMMON_TOKEN",
    "TAP",
    "TRANSITION",
    "UNTAP"
]

# =============================================================================
# Unified Action Types (Command + Action Fusion)
# =============================================================================
# This list combines CommandTypes and EditorActionTypes for the Unified UI.
# We prioritize CommandTypes where overlap exists.
UNIFIED_ACTION_TYPES = sorted(list(set(COMMAND_TYPES + EDITOR_ACTION_TYPES)))
if "NONE" in UNIFIED_ACTION_TYPES:
    UNIFIED_ACTION_TYPES.remove("NONE")
    UNIFIED_ACTION_TYPES.insert(0, "NONE")


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
    "CANNOT_ATTACK_OR_BLOCK",
    "CANNOT_ATTACK_AND_BLOCK"
]

# Keywords that can be set via SET_KEYWORD (subset or extension of grantable)
# For now, same as GRANTABLE_KEYWORDS but can be extended for permanent effects
SETTABLE_KEYWORDS = GRANTABLE_KEYWORDS.copy()

# =============================================================================
# Trigger Types (Effect Triggers)
# =============================================================================
TRIGGER_TYPES = [
    "ON_PLAY",
    "ON_ATTACK",
    "ON_BLOCK",
    "ON_DESTROY",
    "TURN_START",
    "PASSIVE_CONST",
    "ON_OTHER_ENTER",
    "ON_ATTACK_FROM_HAND",
    "AT_BREAK_SHIELD",
    "ON_CAST_SPELL",
    "ON_OPPONENT_DRAW",
    "ON_OPPONENT_CREATURE_ENTER"
]

# Triggers valid for Spells (subset of above)
SPELL_TRIGGER_TYPES = [
    "ON_PLAY",
    "ON_CAST_SPELL",
    "TURN_START",
    "ON_OPPONENT_DRAW",
    "PASSIVE_CONST",
    "ON_OTHER_ENTER"
]

# =============================================================================
# Layer Types (Static Modifiers)
# =============================================================================
LAYER_TYPES = [
    "COST_MODIFIER",
    "POWER_MODIFIER",
    "GRANT_KEYWORD",
    "SET_KEYWORD",
    "FORCE_ATTACK"
]

# =============================================================================
# Game Results
# =============================================================================
GAME_RESULTS = [
    "WIN",
    "LOSE",
    "DRAW"
]

# =============================================================================
# Query Modes
# =============================================================================
QUERY_MODES = [
    "CARDS_MATCHING_FILTER",
    "MANA_COUNT",
    "CREATURE_COUNT",
    "SHIELD_COUNT",
    "HAND_COUNT",
    "GRAVEYARD_COUNT",
    "BATTLE_ZONE_COUNT",
    "OPPONENT_MANA_COUNT",
    "OPPONENT_CREATURE_COUNT",
    "OPPONENT_SHIELD_COUNT",
    "OPPONENT_HAND_COUNT",
    "OPPONENT_GRAVEYARD_COUNT",
    "OPPONENT_BATTLE_ZONE_COUNT",
    "CARDS_DRAWN_THIS_TURN",
    "MANA_CIVILIZATION_COUNT"
]

# =============================================================================
# Duration Types
# =============================================================================
DURATION_TYPES = [
    "THIS_TURN",
    "UNTIL_START_OF_YOUR_TURN",
    "UNTIL_END_OF_YOUR_TURN",
    "UNTIL_START_OF_OPPONENT_TURN",
    "UNTIL_END_OF_OPPONENT_TURN",
    "PERMANENT",
    "DURING_OPPONENT_TURN"
]
