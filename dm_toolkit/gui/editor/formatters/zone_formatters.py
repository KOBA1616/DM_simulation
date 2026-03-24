from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.gui.i18n import tr

@register_formatter("REVEAL_TO_BUFFER")
class RevealToBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        src_zone = tr(command.get("from_zone", "DECK"))
        val1 = get_command_amount(command, default=0)
        amt = val1 if val1 > 0 else 1
        optional = bool(command.get("optional", False))
        verb = "置いてもよい。" if optional else "置く。"
        return f"{src_zone}から{amt}枚を表向きにしてバッファに{verb}"

@register_formatter("MOVE_BUFFER_TO_ZONE")
class MoveBufferToZoneFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        to_zone = tr(command.get("to_zone", "HAND"))
        filter_def = command.get("filter") or command.get("target_filter") or {}
        civs = filter_def.get("civilizations", []) if filter_def else []
        types = filter_def.get("types", []) if filter_def else []
        races = filter_def.get("races", []) if filter_def else []

        has_filter = bool(civs or types or races)
        val1 = get_command_amount(command, default=0)
        up_to = bool(command.get('up_to', False))

        if has_filter:
            civ_part = ""
            if civs:
                civ_part = "/".join(CardTextResources.get_civilization_text(c) for c in civs) + "の"

            if races:
                type_part = "/".join(races)
            elif "ELEMENT" in types:
                type_part = "エレメント"
            elif "SPELL" in types and "CREATURE" not in types:
                type_part = "呪文"
            elif "CREATURE" in types:
                type_part = "クリーチャー"
            else:
                type_part = "カード"

            qty_part = f"{val1}枚" if val1 > 0 else "すべて"
            if val1 > 0 and up_to:
                 qty_part = f"最大{val1}枚"

            optional = bool(command.get("optional", False))
            verb = "加えてもよい。" if optional else "加える。"
            return f"その中から、{civ_part}{type_part}を{qty_part}選び、{to_zone}に{verb}"
        else:
            qty_part = f"{val1}枚" if val1 > 0 else "すべて"
            if val1 > 0 and up_to:
                qty_part = f"最大{val1}枚"
            optional = bool(command.get("optional", False))
            verb = "加えてもよい。" if optional else "加える。"
            return f"その中から、{qty_part}を{to_zone}に{verb}"

@register_formatter("REPLACE_CARD_MOVE")
class ReplaceCardMoveFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        dest_zone = command.get("destination_zone", "")
        if not dest_zone:
            dest_zone = command.get("to_zone", "DECK_BOTTOM")

        src_zone = command.get("source_zone", "")
        if not src_zone:
            src_zone = command.get("from_zone", "GRAVEYARD")

        zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
        orig_zone_str = CardTextResources.get_zone_text(src_zone) if src_zone else "元のゾーン"
        up_to_flag = bool(command.get('up_to', False))

        scope = command.get("target_group") or command.get("scope", "NONE")
        is_self_ref = scope == "SELF"

        input_key = command.get("input_value_key") or command.get("input_link") or ""

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

        if input_key:
            input_usage = str(command.get("input_value_usage") or command.get("input_usage") or "").upper()
            link_suffix = InputLinkFormatter.format_input_link_context_suffix(command)
            linked_target = "そのカード"
            if input_usage == "REPLACEMENT":
                 if is_self_ref:
                     return f"かわりに、{orig_zone_str}に置くかわりに{zone_str}に置く。"
                 else:
                     return f"かわりに、{orig_zone_str}に置くかわりに{zone_str}に置く。"

            return f"その後、{linked_target}を{orig_zone_str}に置くかわりに{zone_str}に置く。"

        val1 = get_command_amount(command, default=0)

        # Build target text explicitly instead of using template.
        target_str, unit = cls._resolve_target(command, ctx.is_spell)

        if is_self_ref:
             t = f"このカードを{orig_zone_str}に置くかわりに{zone_str}に置く。"
        else:
             if val1 > 0:
                  qty = f"最大{val1}{unit}" if up_to_flag else f"{val1}{unit}"
                  t = f"{target_str}を{qty}選び、{orig_zone_str}に置くかわりに{zone_str}に置く。"
             else:
                  t = f"{target_str}を{orig_zone_str}に置くかわりに{zone_str}に置く。"

        return t
