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

        # AST/Context integration to detect if this command is structurally composite
        is_composite_action = False
        if ctx and hasattr(ctx, "current_commands_list") and ctx.current_commands_list:
            # Context-aware logic can determine if we're evaluating inside a sequence
            # where "その後" is prevalent, enabling smarter end-of-sentence decisions.
            is_composite_action = len(ctx.current_commands_list) > 1

        return TextUtils.apply_conjugation(text, optional, is_composite_action=is_composite_action)


    @classmethod
    def _resolve_player_noun(cls, command: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        """
        Helper to centrally resolve the player noun for a command.
        Uses target_group or scope, falls back to standard target resolution.
        """
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
        scope = command.get("target_group") or command.get("scope", "NONE")
        noun = TargetResolutionService.resolve_noun(scope)
        if noun:
            return noun
        target_str, _ = cls._resolve_target(command, ctx)
        return target_str

    @classmethod
    def _resolve_target(cls, action: Dict[str, Any], ctx: TextGenerationContext = None, default_self_noun: str = "", **kwargs) -> tuple[str, str]:
        """
        Helper to delegate target resolution to TargetFormatter.
        """
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
        return TargetResolutionService.format_target(action, ctx, default_self_noun=default_self_noun, **kwargs)
