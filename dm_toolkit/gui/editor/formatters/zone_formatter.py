from typing import List
from dm_toolkit.gui.editor.text_resources import CardTextResources

class ZoneFormatter:
    """Formatter for handling lists of zones and their corresponding context particles."""

    @classmethod
    def format_zone_list(cls, zones: List[str], context: str = "in", joiner: str = "、または") -> str:
        """
        Formats a list of zones with the appropriate joining phrase and context particle.

        Args:
            zones: A list of zone strings (e.g., ['HAND', 'MANA_ZONE']).
            context: The grammatical context for the particle ('from' -> 'から', 'to' -> 'に', 'in' -> 'の').
                     If context is an empty string, no trailing particle is added.
            joiner: The string used to join multiple zones (e.g., 'と', '、または').
        """
        if not zones:
            return ""

        # Use CardTextResources to get localized text and handle standard '、または' joining
        # if the joiner matches what format_zones_list uses by default.
        # But we want to explicitly support custom joiners like "と".

        # We handle single zone or standard multi-zone directly
        formatted_zones = []
        for z in zones:
            normalized = CardTextResources.normalize_zone_name(z)
            if normalized:
                text = CardTextResources.get_zone_text(normalized)
                if text:
                    formatted_zones.append(text)

        if not formatted_zones:
             return ""

        base_text = joiner.join(formatted_zones)

        # Apply particle based on context
        if context == "from":
            return f"{base_text}から"
        elif context == "to":
            return f"{base_text}に"
        elif context == "in":
            return f"{base_text}の"

        return base_text
