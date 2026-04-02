# -*- coding: utf-8 -*-
from typing import Dict, Any
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.consts import TimingMode

class TriggerFormatter:
    @classmethod
    def apply_trigger_scope(
        cls,
        trigger_text: str,
        scope: str,
        trigger_type: str,
        trigger_filter: Dict[str, Any] = None,
        timing_mode: str = TimingMode.POST.value,
    ) -> str:
        """
        Apply scope prefix to trigger text (e.g., "ON_CAST_SPELL" + "OPPONENT" -> "相手が呪文を唱えた時").
        """
        if not scope or scope == "NONE" or scope == "ALL":
            return trigger_text

        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService as TargetScopeResolver
        scope_text = TargetScopeResolver.resolve_noun(scope)
        if not scope_text:
            return trigger_text

        from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
        # Uses FilterTextFormatter to avoid scope prefix duplication
        # if the trigger text already has "相手が" or "自分が", etc.
        formatted = FilterTextFormatter.format_scope_prefix(scope, trigger_text)
        # If FilterTextFormatter actually did not prepend anything new (it detected the prefix),
        # return the trigger text as-is to preserve structural replacements correctly,
        # or we could just use its output which is identical.
        if formatted == trigger_text:
            return trigger_text

        # Structured template composition (timing/scope/trigger as variables)
        tmpl_set = CardTextResources.TRIGGER_COMPOSITION_TEMPLATES.get(trigger_type)
        if tmpl_set:
            timing_key = str(timing_mode or "").upper()
            if timing_key not in (TimingMode.PRE.value, TimingMode.POST.value):
                timing_key = TimingMode.POST.value
            tmpl = tmpl_set.get(timing_key) or tmpl_set.get(TimingMode.POST.value)
            if tmpl:
                from dm_toolkit.consts import CardType
                default_type = CardType.SPELL.value if trigger_type == "ON_CAST_SPELL" else CardType.CREATURE.value
                subject = TargetScopeResolver.compose_subject_from_filter(trigger_filter, default_type)
                return tmpl.format(scope_text=scope_text, subject=subject)

        # For fallback triggers without specific COMPOSITION_TEMPLATES,
        # apply replacement phrases directly if PRE timing mode is requested,
        # ensuring tests for legacy trigger formatting (like ON_OPPONENT_CREATURE_ENTER) pass without guessing.
        if timing_mode == TimingMode.PRE.value:
            trigger_text = cls.to_replacement_trigger_text(trigger_text)

        # Default fallbacks
        if trigger_text.startswith("この"):
             # "このクリーチャー..." -> "相手のこのクリーチャー..." (Syntactically valid for 'Target's this creature')
             return f"{scope_text}の{trigger_text}"

        # Default to "の" prefix
        return f"{scope_text}の{trigger_text}"

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
        from dm_toolkit.gui.editor.formatters.command_registry import CommandFormatterRegistry
        from dm_toolkit.gui.editor.formatters.clause_joiner import ClauseJoiner

        target_str, _ = cls._resolve_target(command, ctx)
        trigger_cmd = command.get("replaced_action", {})
        action_cmd = command.get("replacement_action", {})
        trigger_condition_ast = command.get("trigger_condition", {})

        # Try evaluating trigger_condition AST first if provided natively
        if trigger_condition_ast:
             from_text = ClauseJoiner.join_condition_ast(trigger_condition_ast, ctx, is_root=True)
             if from_text.endswith("なら、"):
                 from_text = from_text.removesuffix("なら、") + "時"
             elif from_text.endswith("、"):
                 from_text = from_text.removesuffix("、") + "時"
             elif from_text.endswith("。"):
                 from_text = from_text.removesuffix("。") + "時"
        elif trigger_cmd:
            from_text = CommandFormatterRegistry.format_command(trigger_cmd, ctx)
        else:
            from_text = f"{target_str}が破壊される時" # fallback

        # Ensure it sounds like a future condition "する時"
        from_text = TriggerFormatter.to_replacement_trigger_text(from_text)

        # Format the replacement action. Note that replacing action may be an AST structure/list
        if action_cmd:
            to_text = CommandFormatterRegistry.format_command(action_cmd, ctx)
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
