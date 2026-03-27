from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.gui.editor.formatters.legacy_action_formatters import LegacyActionFormatterHelper
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.i18n import tr

@register_formatter("TRANSITION")
class TransitionFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        input_key = command.get("input_value_key", "")
        from_z = CardTextResources.normalize_zone_name(command.get("from_zone", ""))
        to_z = CardTextResources.normalize_zone_name(command.get("to_zone", ""))
        amount = get_command_amount(command, default=0)
        up_to_flag = bool(command.get('up_to', False))

        target_str, unit = cls._resolve_target(command, ctx.is_spell)

        template_key = (from_z, to_z)
        if template_key in CardTextResources.ZONE_MOVE_TEMPLATES:
            template = CardTextResources.ZONE_MOVE_TEMPLATES[template_key]

            if template_key == ("BATTLE_ZONE", "GRAVEYARD") or template_key == ("BATTLE_ZONE", "MANA_ZONE"):
                if up_to_flag and amount > 0:
                    template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
                elif amount == 0 and not input_key:
                    template = "{from_z}の{target}をすべて{to_z}に置く。"
            elif template_key == ("BATTLE_ZONE", "HAND"):
                if up_to_flag and amount > 0:
                    template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に戻す。"
                elif amount == 0 and not input_key:
                    template = "{from_z}の{target}をすべて{to_z}に戻す。"
            elif template_key == ("HAND", "MANA_ZONE"):
                if up_to_flag and amount > 0:
                    template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
            elif template_key == ("DECK", "HAND"):
                if up_to_flag:
                    if target_str != "カード":
                        template = "{from_z}から{target}を{amount}{unit}まで選び、{to_z}に加える。"
                    else:
                        template = "山札からカードを最大{amount}枚まで選び、手札に加える。"
                else:
                    if target_str != "カード":
                        template = "{from_z}から{target}を{amount}{unit}選び、{to_z}に加える。"
            elif template_key == ("GRAVEYARD", "HAND"):
                if up_to_flag and amount > 0:
                    template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に戻す。"
            elif template_key == ("GRAVEYARD", "BATTLE_ZONE"):
                if up_to_flag and amount > 0:
                    template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に出す。"
        else:
            if to_z == "GRAVEYARD":
                if up_to_flag and amount > 0:
                    template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
                else:
                    template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
            elif to_z == "DECK_BOTTOM":
                if input_key:
                    normalized_from = CardTextResources.normalize_zone_name(from_z)
                    scope = command.get("target_group") or command.get("scope", "NONE")
                    if normalized_from == "HAND":
                        to_zone_text = CardTextResources.get_zone_text(to_z)
                        linked_count = InputLinkFormatter.format_linked_count_token(command, "その同じ数")
                        owner = ""
                        if scope in ["PLAYER_SELF", "SELF"]:
                            owner = "自分の"
                        elif scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                            owner = "相手の"
                        elif scope == "ALL_PLAYERS":
                            owner = "各プレイヤーの"
                        if up_to_flag:
                            template = f"{owner}手札から{{target}}を{linked_count}だけまで選び、{to_zone_text}に置く。"
                        else:
                            template = f"{owner}手札から{{target}}を{linked_count}だけ選び、{to_zone_text}に置く。"
                    elif up_to_flag:
                        template = "{from_z}の{target}をその同じ数だけまで選び、{to_z}に置く。"
                    else:
                        template = "{from_z}の{target}をその同じ数だけ選び、{to_z}に置く。"
                else:
                    if up_to_flag and amount > 0:
                        template = "{from_z}の{target}を{amount}{unit}まで選び、{to_z}に置く。"
                    else:
                        template = "{from_z}の{target}を{amount}{unit}{to_z}に置く。"
            else:
                template = CardTextResources.ZONE_MOVE_TEMPLATES.get("DEFAULT", "{target}を{from_z}から{to_z}へ移動する。")

        if "{from_z}" in template:
            template = template.replace("{from_z}", CardTextResources.get_zone_text(from_z))
        if "{to_z}" in template:
            template = template.replace("{to_z}", CardTextResources.get_zone_text(to_z))

        template = template.replace("{amount}", str(amount))

        text = LegacyActionFormatterHelper.apply_replacements(command, ctx, template, str(amount), target_str, unit)
        return LegacyActionFormatterHelper.apply_conjugation(command, text)

@register_formatter("MOVE_CARD")
class MoveCardFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        dest_zone = command.get("destination_zone") or command.get("to_zone", "")
        input_key = command.get("input_value_key", "")
        amount = get_command_amount(command, default=0)
        is_all = (amount == 0 and not input_key)
        up_to_flag = bool(command.get('up_to', False))

        src_zone = command.get("source_zone", "")
        src_str = tr(src_zone) if src_zone else ""
        zone_str = tr(dest_zone) if dest_zone else "どこか"

        target_str, unit = cls._resolve_target(command, ctx.is_spell)

        if dest_zone == "HAND":
            if up_to_flag and amount > 0:
                template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に戻す。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に戻す。")
            else:
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に戻す。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に戻す。")
            if is_all:
                template = (f"{{target}}をすべて{zone_str}に戻す。" if not src_str
                            else f"{src_str}の{{target}}をすべて{zone_str}に戻す。")
        elif dest_zone == "MANA_ZONE":
            if up_to_flag and amount > 0:
                template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。")
            else:
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
            if is_all:
                template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}をすべて{zone_str}に置く。")
        elif dest_zone == "GRAVEYARD":
            if up_to_flag and amount > 0:
                template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。")
            else:
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
            if is_all:
                template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}をすべて{zone_str}に置く。")
        elif dest_zone == "DECK_BOTTOM":
            if up_to_flag and amount > 0:
                template = (f"{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}まで選び、{zone_str}に置く。")
            else:
                template = (f"{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}を{{value1}}{{unit}}選び、{zone_str}に置く。")
            if is_all:
                template = (f"{{target}}をすべて{zone_str}に置く。" if not src_str
                            else f"{src_str}の{{target}}をすべて{zone_str}に置く。")
        else:
            template = ""

        if template:
            text = LegacyActionFormatterHelper.apply_replacements(command, ctx, template, str(amount), target_str, unit)
            return LegacyActionFormatterHelper.apply_conjugation(command, text)
        return ""

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
