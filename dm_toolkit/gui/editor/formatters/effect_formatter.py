from typing import Dict, Any
from dm_toolkit.gui.editor.formatters.abilities.static_ability_formatter import StaticAbilityFormatter
from dm_toolkit.gui.editor.formatters.abilities.triggered_ability_formatter import TriggeredAbilityFormatter
from dm_toolkit.gui.editor.formatters.modifier_formatters import ModifierFormatterRegistry
from dm_toolkit.gui.editor.formatters.condition_formatter import ConditionFormatter
from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
from dm_toolkit.consts import TargetScope

class EffectFormatter:
    """Centralized dispatch method for formatting effects and modifiers."""

    @classmethod
    def format_modifier(cls, modifier: Dict[str, Any], ctx: Any = None) -> str:
        mtype = modifier.get("type", "NONE")
        condition = modifier.get("condition", {})
        filter_def = modifier.get("filter", {})
        value = modifier.get("value", 0)

        scope = modifier.get("scope", TargetScope.ALL)
        scope = TargetScope.normalize(scope)

        if ctx and hasattr(ctx, "error_reporter"):
            with ctx.error_reporter.path_segment("condition"):
                cond_text = ConditionFormatter.format_condition_text(condition, ctx)
        else:
            cond_text = ConditionFormatter.format_condition_text(condition, ctx)

        if mtype:
            ModifierFormatterRegistry.update_metadata(mtype, modifier, ctx)

        full_target = TargetResolutionService.format_modifier_target(filter_def, scope=scope)
        return ModifierFormatterRegistry.format(mtype, cond_text, full_target, value, modifier, ctx)

    @classmethod
    def format_effect(cls, effect: Dict[str, Any], ctx: Any) -> str:
        if isinstance(effect, dict):
            effect_type = effect.get("type", "")
            trigger = effect.get("trigger", "NONE")

            if effect_type in ("COST_MODIFIER", "POWER_MODIFIER", "GRANT_KEYWORD", "SET_KEYWORD", "ADD_RESTRICTION"):
                if trigger == "NONE" or trigger not in effect:
                    return cls.format_modifier(effect, ctx=ctx)

        trigger = effect.get("trigger", "NONE")
        triggers = effect.get("triggers", [])
        if not triggers and trigger != "NONE":
            triggers = [trigger]

        condition = effect.get("condition", {}) or {}

        if ctx and hasattr(ctx, "error_reporter"):
            with ctx.error_reporter.path_segment("condition"):
                cond_text = ConditionFormatter.format_condition_text(condition, ctx)
        else:
            cond_text = ConditionFormatter.format_condition_text(condition, ctx)

        has_active_trigger = any(t not in ("NONE", "PASSIVE_CONST") for t in triggers)
        is_passive = any(t == "PASSIVE_CONST" for t in triggers) and not has_active_trigger

        if is_passive or trigger == "PASSIVE_CONST" or not has_active_trigger:
            return StaticAbilityFormatter.format(effect, ctx, cond_text)
        else:
            return TriggeredAbilityFormatter.format(effect, ctx, cond_text)
