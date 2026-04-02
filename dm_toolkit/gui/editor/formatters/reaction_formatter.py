from typing import Dict, Any
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.i18n import tr

class ReactionFormatter:
    """Formatter for reaction abilities (e.g., S-Trigger, G-Strike)."""

    @classmethod
    def format(cls, reaction: Dict[str, Any]) -> str:
        if not reaction:
            return ""
        rtype = reaction.get("type", "NONE")

        formatter = CardTextResources.REACTION_TEXT_MAP.get(rtype)
        if formatter is not None:
            return formatter(reaction)

        return tr(rtype)
