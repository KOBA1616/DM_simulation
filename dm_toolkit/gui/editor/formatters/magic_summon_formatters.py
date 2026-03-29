from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.consts import MAX_COST_VALUE

@register_formatter("CAST_SPELL")
class CastSpellFormatter(CommandFormatterBase):
    @classmethod
    def _format_cast_spell_cost_phrase(cls, action: Dict[str, Any]) -> str:
        """Return the cost phrase used by CAST_SPELL preview text."""
        play_flags = action.get("play_flags")
        explicit_cost = action.get("cost")

        is_free = True
        if isinstance(play_flags, bool):
            is_free = play_flags
        elif isinstance(play_flags, list):
            is_free = "FREE" in play_flags or "COST_FREE" in play_flags
        elif explicit_cost not in (None, 0):
            is_free = False

        if is_free:
            return "コストを支払わずに唱える"
        if isinstance(explicit_cost, int) and explicit_cost > 0:
            return f"コスト{explicit_cost}を支払って唱える"
        return "コストを支払って唱える"

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        action = command.copy()
        temp_filter = action.get("filter") or action.get("target_filter") or {}
        if not isinstance(temp_filter, dict):
            temp_filter = {}
        temp_filter = temp_filter.copy()
        action["filter"] = temp_filter

        is_mega_last_burst = action.get("is_mega_last_burst", False) or action.get("mega_last_burst", False) or ctx.has_mega_last_burst
        mega_burst_prefix = "このクリーチャーがバトルゾーンから離れて、" if is_mega_last_burst else ""
        cast_phrase = cls._format_cast_spell_cost_phrase(action)

        input_usage = str(action.get("input_value_usage") or action.get("input_usage") or "").upper()

        usage_label_suffix = ""
        linked_text = InputLinkFormatter.resolve_linked_value_text(action, context_commands=ctx.current_commands_list)
        if linked_text and input_usage:
            label = InputLinkFormatter.format_input_usage_label(input_usage)
            if label:
                usage_label_suffix = f"（{label}）"

        types = temp_filter.get("types", [])
        if "SPELL" in types or not types:
            zones = temp_filter.get("zones", [])
            linked_cost_phrase = ""
            max_cost_def = temp_filter.get("max_cost")
            if InputLinkFormatter.is_input_linked(max_cost_def, usage="MAX_COST"):
                source_token = InputLinkFormatter.format_linked_count_token(action, "その数")
                source_token = InputLinkFormatter.normalize_linked_count_label(source_token)
                linked_cost_phrase = f"{source_token}以下のコストの"

            zone_phrase = ""
            if zones:
                zone_phrase = CardTextResources.format_zones_list(zones) + "から"

            val1 = action.get("value1", 0)
            if not val1 and "count" in temp_filter:
                val1 = temp_filter.get("count", 0)

            if val1 > 0:
                return f"{mega_burst_prefix}{zone_phrase}{linked_cost_phrase}呪文を{val1}枚まで{cast_phrase}。{usage_label_suffix}"
            elif linked_cost_phrase:
                return f"{mega_burst_prefix}{zone_phrase}{linked_cost_phrase}呪文を{cast_phrase}。{usage_label_suffix}"

        target_str, unit = cls._resolve_target(action, ctx)
        if target_str == "" or target_str == "カード":
            return f"{mega_burst_prefix}カードを{cast_phrase}。{usage_label_suffix}"
        else:
            return f"{mega_burst_prefix}{target_str}を{cast_phrase}。{usage_label_suffix}"

@register_formatter("PLAY_FROM_BUFFER")
class PlayFromBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx)
        return f"選んだカード（{target_str}）を使う。"

@register_formatter("ADD_MANA")
class AddManaFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        ctx.metadata["mana_charged"] = True

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        val1 = get_command_amount(command, default=0)
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)
        if linked_text:
             return f"自分の山札の上から、{linked_text}だけタップしてマナゾーンに置く。"
        return f"自分の山札の上から{val1}枚をタップしてマナゾーンに置く。"

@register_formatter("CHOICE")
class ChoiceFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        val1 = get_command_amount(command, default=0)
        flags = command.get("flags", []) or []
        optional = False
        if isinstance(flags, list) and "ALLOW_DUPLICATES" in flags:
            optional = True

        options = command.get("options", [])
        if not options:
            return ""

        parts = []
        for opt in options:
             if isinstance(opt, list):
                  opt_parts = []
                  for a in opt:
                       if isinstance(a, dict):
                            opt_parts.append(CardTextGenerator._format_command(a, ctx))
                  chain_text = " ".join(opt_parts)
                  parts.append(f"> {chain_text}")
             elif isinstance(opt, dict):
                  parts.append(f"> {CardTextGenerator._format_command(opt, ctx)}")

        lines = []
        if len(parts) > 0:
            qty_text = "どれか1つ" if val1 == 1 else f"{val1}回"
            lines.append(f"次の中から{qty_text}選ぶ。")
            lines.extend(parts)

        return "\n".join(lines)
