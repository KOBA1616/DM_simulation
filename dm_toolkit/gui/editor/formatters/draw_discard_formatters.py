from typing import Dict, Any, List, Optional
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils

@register_formatter("DRAW_CARD")
class DrawCardFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        ctx.metadata["draws"] = True

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        up_to = bool(command.get('up_to', False))
        template = CardTextResources.ACTION_MAP.get("DRAW_CARD", "")
        if up_to:
            template = "最大{value1}枚まで引く。"



        # Map 'amount' to 'value1' since command dictionary uses 'amount' typically
        val1 = get_command_amount(command, default=0)

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)
        if linked_text:
            return template.replace("{value1}", linked_text)

        # Default target_str for DRAW_CARD (not using {target})
        return template.replace("{value1}", str(val1)).replace("{target}", "カード")

@register_formatter("DISCARD")
class DiscardFormatter(CommandFormatterBase):
    @classmethod
    def update_metadata(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> None:
        ctx.metadata["discards"] = True

    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        scope = command.get("target_group") or command.get("scope", "NONE")
        if scope in ["PLAYER_SELF", "SELF"]:
            scope = "NONE"
            command["scope"] = scope

        up_to = bool(command.get('up_to', False))
        val1 = get_command_amount(command, default=0)

        target_str, unit = cls._resolve_target(command, getattr(ctx, "is_spell", False))
        cnt = val1
        tgt = target_str

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)
        if linked_text:
             cnt = linked_text
             unit = "枚" # assuming cards

        if cnt == 0 and not linked_text:
            if scope == "NONE":
                return "手札をすべて捨てる。"
            else:
                return f"{tgt}をすべて捨てる。"

        if not tgt or tgt == "カード" or tgt == "自分のカード" or tgt == "自分の":
            tgt = "手札"

        if up_to:
            text = f"{tgt}を最大{cnt}{unit}捨てる。"
        else:
            text = f"{tgt}を{cnt}{unit}捨てる。"

        return text
