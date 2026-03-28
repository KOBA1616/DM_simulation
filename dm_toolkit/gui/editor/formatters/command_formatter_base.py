from typing import Dict, Any, List, Optional
import dm_toolkit.consts as consts
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils

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
    def format_with_optional(cls, command: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        """
        Formats the command and automatically applies optional conjugation if needed.
        """
        text = cls.format(command, ctx)
        optional = bool(command.get("optional", False))
        return TextUtils.apply_conjugation(text, optional)

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], is_spell: bool = False, default_self_noun: str = "", **kwargs) -> tuple[str, str]:
        """
        Helper to delegate target resolution to TargetFormatter.
        """
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
        return TargetResolutionService.format_target(action, is_spell, default_self_noun=default_self_noun, **kwargs)
