from typing import Dict, Any, List, Optional
import dm_toolkit.consts as consts
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

class CommandFormatterBase:
    """Base class for all command text formatters."""

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        """
        Generate Japanese text for the given command.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement format()")

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], is_spell: bool = False, **kwargs) -> tuple[str, str]:
        """
        Helper to delegate target resolution to TargetFormatter.
        """
        from dm_toolkit.gui.editor.formatters.target_formatter import TargetFormatter
        return TargetFormatter.format_target(action, is_spell, **kwargs)
