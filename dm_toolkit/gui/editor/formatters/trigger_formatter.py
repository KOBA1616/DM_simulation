# -*- coding: utf-8 -*-
from typing import Dict, Any
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.consts import TimingMode

class TriggerFormatter:
    @classmethod
    def resolve_effect_timing_mode(cls, effect: Dict[str, Any]) -> str:
        """Normalize effect timing mode for text composition."""
        if not isinstance(effect, dict):
            return TimingMode.POST.value
        mode = str(effect.get("timing_mode", "") or "").upper()
        if mode in (TimingMode.PRE.value, TimingMode.POST.value):
            return mode
        return TimingMode.PRE.value if cls.is_replacement_effect(effect) else TimingMode.POST.value

    @classmethod
    def to_replacement_trigger_text(cls, trigger_text: str) -> str:
        """Convert post-event trigger text (〜た時) into replacement tone (〜る時)."""
        text = trigger_text
        for src, dst in CardTextResources.TRIGGER_REPLACEMENT_MAP:
            if src in text:
                return text.replace(src, dst)
        return text

    @classmethod
    def is_replacement_effect(cls, effect: Dict[str, Any]) -> bool:
        """Return True if the effect should be rendered as PRE/replacement timing."""
        if not isinstance(effect, dict):
            return False
        return effect.get("mode") == "REPLACEMENT" or effect.get("timing_mode") == TimingMode.PRE.value

    @classmethod
    def trigger_to_japanese(cls, trigger: str, is_spell: bool = False, effect: Dict[str, Any] = None) -> str:
        """Get Japanese trigger text, applying replacement phrasing when needed."""
        base = CardTextResources.get_trigger_text(trigger, is_spell=is_spell)
        if effect is not None and cls.is_replacement_effect(effect):
            return cls.to_replacement_trigger_text(base)

        return base

from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext

@register_formatter("REPLACEMENT_EFFECT")
class ReplacementEffectFormatter(CommandFormatterBase):
    """Formatter to strictly decouple and structure Replacement Effects."""

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        target_str, _ = cls._resolve_target(command, ctx)
        trigger_cmd = command.get("replaced_action", {})
        action_cmd = command.get("replacement_action", {})

        # Format the original action that is being replaced (e.g., destroy)
        if trigger_cmd:
            from_text = CardTextGenerator.format_command(trigger_cmd, ctx)
        else:
            from_text = f"{target_str}が破壊される時" # fallback

        # Ensure it sounds like a future condition "する時"
        from_text = TriggerFormatter.to_replacement_trigger_text(from_text)

        # Format the replacement action
        if action_cmd:
            to_text = CardTextGenerator.format_command(action_cmd, ctx)
        else:
            to_text = "何もしない。" # fallback

        return cls.format_string(from_text, to_text)

    @classmethod
    def format_string(cls, trigger_text: str, actions_text: str) -> str:
        """Utility for string-based replacements when parsing top-level effect structures."""
        if not trigger_text:
            return actions_text
        if trigger_text.endswith("、"):
            return f"{trigger_text}かわりに{actions_text}"
        return f"{trigger_text}、かわりに{actions_text}"
