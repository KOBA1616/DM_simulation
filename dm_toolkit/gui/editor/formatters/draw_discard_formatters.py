from typing import Dict, Any, List, Optional
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources

@register_formatter("DRAW_CARD")
class DrawCardFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        up_to = bool(command.get('up_to', False))
        optional = bool(command.get("optional", False))

        if optional:
            template = CardTextResources.ACTION_OPTIONAL_MAP.get("DRAW_CARD", "")
            if up_to:
                template = "最大{value1}枚まで引いてもよい。"
        else:
            template = CardTextResources.ACTION_MAP.get("DRAW_CARD", "")
            if up_to:
                template = "最大{value1}枚まで引く。"

        # Map 'amount' to 'value1' since command dictionary uses 'amount' typically
        val1 = command.get("amount") if command.get("amount") is not None else command.get("value1", 0)

        # Check input_value_key
        input_key = command.get("input_value_key") or command.get("input_link") or ""

        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        if input_key:
            input_label = command.get("_input_value_label", "")
            if not input_label:
                 input_label = CardTextGenerator._format_input_source_label(command)
            if input_label:
                 val_str = f"その数"
                 return template.replace("{value1}", val_str)

        # Default target_str for DRAW_CARD (not using {target})
        return template.replace("{value1}", str(val1)).replace("{target}", "カード")

@register_formatter("DISCARD")
class DiscardFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], is_spell: bool = False, sample: List[Any] = None, card_mega_last_burst: bool = False) -> str:
        scope = command.get("target_group") or command.get("scope", "NONE")
        if scope in ["PLAYER_SELF", "SELF"]:
            scope = "NONE"
            command["scope"] = scope

        up_to = bool(command.get('up_to', False))
        val1 = command.get("amount") if command.get("amount") is not None else command.get("value1", 0)

        target_str, unit = cls._resolve_target(command, is_spell)
        cnt = val1
        tgt = target_str

        input_key = command.get("input_value_key") or command.get("input_link") or ""
        if input_key:
            from dm_toolkit.gui.editor.text_generator import CardTextGenerator
            input_label = command.get("_input_value_label", "")
            if not input_label:
                 input_label = CardTextGenerator._format_input_source_label(command)
            if input_label:
                 cnt = f"その数"
                 unit = "枚" # assuming cards

        if cnt == 0 and not input_key:
            if scope == "NONE":
                return "手札をすべて捨てる。"
            else:
                return f"{tgt}をすべて捨てる。"

        optional = bool(command.get("optional", False))

        if not tgt or tgt == "カード" or tgt == "自分のカード" or tgt == "自分の":
            tgt = "手札"

        verb = "捨ててもよい。" if optional else "捨てる。"

        if up_to:
            return f"{tgt}を最大{cnt}{unit}{verb}"
        return f"{tgt}を{cnt}{unit}{verb}"
