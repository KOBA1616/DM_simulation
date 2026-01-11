# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional

# Constants defining the universe of possibilities
ALL_TRIGGERS = [
    "ON_PLAY", "ON_ATTACK", "ON_DESTROY", "ON_BLOCK", "ON_OPPONENT_DRAW",
    "S_TRIGGER", "TURN_START", "AT_END_OF_TURN", "PASSIVE_CONST"
]

SPELL_TRIGGERS = [
    "ON_PLAY", "S_TRIGGER", "PASSIVE_CONST" # Spells usually only have these
]

CREATURE_TRIGGERS = ALL_TRIGGERS

# Keywords that only apply to Creatures
CREATURE_KEYWORDS = [
    "speed_attacker", "blocker", "slayer", "double_breaker", "triple_breaker",
    "world_breaker", "mach_fighter", "power_attacker", "untap_in"
]

# Keywords that can apply to Spells (or generic)
SPELL_KEYWORDS = [
    "shield_trigger", "shield_burn" # S-Trigger is handled via keywords or TriggerType
]

class LogicMaskManager:
    """
    Manages the rules for validity of various Card logic combinations.
    Used to filter UI options based on context (e.g., Card Type).
    """

    @staticmethod
    def get_allowed_triggers(card_type: str) -> List[str]:
        """Returns the list of TriggerTypes valid for the given card type."""
        if card_type == "SPELL":
            return SPELL_TRIGGERS
        return CREATURE_TRIGGERS

    @staticmethod
    def get_allowed_keywords(card_type: str) -> List[str]:
        """Returns the list of Keywords valid for the given card type."""
        if card_type == "SPELL":
            return SPELL_KEYWORDS
        # Default to all for Creatures/Evolution Creatures
        # Merge lists to ensure complete set if needed, or just return a master list
        # For now returning a broad list.
        return CREATURE_KEYWORDS + SPELL_KEYWORDS

    @staticmethod
    def is_action_compatible(action_type: str, card_type: str) -> bool:
        """Checks if a specific Action type is compatible with the card type."""
        # Example: SUMMON_TOKEN might be valid everywhere, but maybe some actions are specific?
        return True

    @staticmethod
    def filter_command_groups(groups: Dict[str, List[str]], card_type: str) -> Dict[str, List[str]]:
        """
        Returns a filtered dictionary of Command Groups -> Types based on card type.
        Currently, most commands are generic, but this provides the hook.
        """
        # Placeholder: No specific filtering yet
        return groups
