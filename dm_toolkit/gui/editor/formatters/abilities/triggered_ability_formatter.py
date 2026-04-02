from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.consts import TimingMode
from dm_toolkit.gui.editor.formatters.input_link_ast import InputLinkASTBuilder
from dm_toolkit.gui.editor.formatters.metadata_flags import SemanticMetadataFlags
from dm_toolkit.gui.editor.formatters.trigger_formatter import ReplacementEffectFormatter
import copy

class TriggeredAbilityFormatter:
    """Formatter specifically for triggered abilities (e.g. ON_PLAY, ON_DESTROY)."""

    @classmethod
    def format(cls, effect: Dict[str, Any], ctx: TextGenerationContext, cond_text: str = "") -> str:
        # Import inside the function to avoid circular imports
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        trigger = effect.get("trigger", "NONE")
        trigger_scope = effect.get("trigger_scope", "NONE")
        timing_mode = CardTextGenerator._resolve_effect_timing_mode(effect)
        condition = effect.get("condition", {}) or {}
        cond_type = condition.get("type", "NONE")

        triggers = effect.get("triggers", [])
        if not triggers and trigger != "NONE":
            triggers = [trigger]

        trigger_texts = []
        for t in triggers:
            t_text = CardTextGenerator.trigger_to_japanese(t, ctx.is_spell, effect=effect)
            if trigger_scope and trigger_scope != "NONE" and t != "PASSIVE_CONST":
                from dm_toolkit.gui.editor.formatters.trigger_formatter import TriggerFormatter
                t_text = TriggerFormatter.apply_trigger_scope(
                    t_text,
                    trigger_scope,
                    t,
                    effect.get("trigger_filter", {}),
                    timing_mode=timing_mode,
                )
            else:
                if timing_mode == TimingMode.PRE.value:
                    t_text = CardTextGenerator._to_replacement_trigger_text(t_text)
            trigger_texts.append(t_text)

        trigger_text = "、または".join(trigger_texts) if trigger_texts else ""

        # Refined natural language logic: structured binding of active conditions + trigger events
        active_condition_prefix = ""
        if trigger != "NONE" and trigger != "PASSIVE_CONST":
            if cond_type == "DURING_YOUR_TURN" or cond_type == "DURING_OPPONENT_TURN":
                active_condition_prefix = cond_text
                cond_text = ""
            elif trigger == "ON_OPPONENT_DRAW" and cond_type == "OPPONENT_DRAW_COUNT":
                val = condition.get("value", 0)
                trigger_text = f"相手がカードを引いた時、{val}枚目以降なら"
                cond_text = ""

        action_texts = []
        raw_items = []
        commands = effect.get("commands", [])
        commands_with_labels = InputLinkASTBuilder.infer_command_labels(commands)

        for i, command in enumerate(commands_with_labels):
            if ctx and hasattr(ctx, "error_reporter"):
                with ctx.error_reporter.path_segment(f"commands[{i}]"):
                    command_for_text = copy.deepcopy(command) if isinstance(command, dict) else command
                    raw_items.append(command_for_text)
                    action_texts.append(CardTextGenerator.format_command(command_for_text, ctx))
            else:
                command_for_text = copy.deepcopy(command) if isinstance(command, dict) else command
                raw_items.append(command_for_text)
                action_texts.append(CardTextGenerator.format_command(command_for_text, ctx))

        full_action_text = CardTextGenerator._merge_action_texts(raw_items, action_texts)

        if ctx.is_spell and trigger == "ON_PLAY" and not ctx.metadata.get(SemanticMetadataFlags.ON_CAST_TRIGGER.value, False):
            trigger_text = ""

        has_active_trigger = any(t not in ("NONE", "PASSIVE_CONST") for t in triggers)

        if trigger_text and has_active_trigger:
            if not full_action_text:
                return ""

            if CardTextGenerator.is_replacement_effect(effect):
                full_text = ReplacementEffectFormatter.format_string(f"{active_condition_prefix}{trigger_text}", f"{cond_text}{full_action_text}")
                return full_text.replace(": ", "、")

            return f"{active_condition_prefix}{trigger_text}: {cond_text}{full_action_text}"

        return f"{cond_text}{full_action_text}"
