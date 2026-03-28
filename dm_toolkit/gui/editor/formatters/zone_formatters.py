from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils
from dm_toolkit.gui.i18n import tr

@register_formatter("TRANSITION")
class TransitionFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from_z = CardTextResources.normalize_zone_name(command.get("from_zone", ""))
        to_z = CardTextResources.normalize_zone_name(command.get("to_zone", ""))
        amount = get_command_amount(command, default=0)
        up_to_flag = bool(command.get('up_to', False))

        target_str, unit = cls._resolve_target(command, ctx.is_spell)

        template_key = (from_z, to_z)
        template = CardTextResources.ZONE_MOVE_TEMPLATES.get(template_key, "")

        if not template:
            # Try generic to_z template if specific from->to not found (wildcard matching)
            # In text_resources.py, `|` separated keys are split into tuples (e.g. ("*", "GRAVEYARD"))
            template = CardTextResources.ZONE_MOVE_TEMPLATES.get(("*", to_z), "")
            if not template:
                template = CardTextResources.ZONE_MOVE_TEMPLATES.get((from_z, "*"), "")

            if not template:
                # If still not found, check for a string key like "*|GRAVEYARD" just in case the JSON was modified
                # but text_resources didn't load it correctly, though the dict split handles it.
                template = CardTextResources.ZONE_MOVE_TEMPLATES.get(f"*|{to_z}", "")
                if not template:
                    template = CardTextResources.ZONE_MOVE_TEMPLATES.get(f"{from_z}|*", "")

                if not template:
                    template = CardTextResources.ZONE_MOVE_TEMPLATES.get("DEFAULT", "{target}を{from_z}から{to_z}へ移動する。")

        # Adjust template for up_to and all
        input_key = command.get('input_value_key') or command.get('input_link')
        is_all = (amount == 0 and not input_key)

        if is_all:
            # Handle "all" case ("すべて")
            if to_z == "HAND" and from_z != "DECK":
                template = template.replace("{amount}{unit}", "すべて").replace("選び、", "")
            elif to_z in ["GRAVEYARD", "MANA_ZONE", "DECK_BOTTOM"]:
                template = template.replace("{amount}{unit}", "すべて").replace("選び、", "")

        elif up_to_flag and amount > 0:
            # Handle "up to" case ("まで選び")
            if to_z == "HAND" and from_z != "DECK":
                template = template.replace("{amount}{unit}", "{amount}{unit}まで").replace("戻す", "選び、戻す")
                if "選び、" not in template:
                    template = template.replace("まで{to_z}", "まで選び、{to_z}")
            elif to_z in ["GRAVEYARD", "MANA_ZONE", "DECK_BOTTOM", "BATTLE_ZONE"]:
                template = template.replace("{amount}{unit}", "{amount}{unit}まで").replace("置く", "選び、置く").replace("出す", "選び、出す")
                if "選び、" not in template:
                     template = template.replace("まで{to_z}", "まで選び、{to_z}")
            elif template_key == ("DECK", "HAND"):
                template = template.replace("{amount}{unit}", "{amount}{unit}まで")

        if template_key == ("DECK", "HAND") and target_str == "カード":
             # Exception for generic deck drawing/searching phrasing
             if up_to_flag:
                  template = "山札からカードを最大{amount}枚まで選び、手札に加える。"
             else:
                  template = "{from_z}から{target}を{amount}{unit}選び、{to_z}に加える。"

        # Handle input_key dynamic text for deck bottom generic returns
        if input_key and to_z == "DECK_BOTTOM":
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


        if "{from_z}" in template:
            template = template.replace("{from_z}", CardTextResources.get_zone_text(from_z))
        if "{to_z}" in template:
            template = template.replace("{to_z}", CardTextResources.get_zone_text(to_z))

        template = template.replace("{amount}", str(amount))

        text = template
        val2 = command.get("value2", 0)
        str_val = command.get("str_param") or command.get("str_val", "")
        dest_zone = command.get("destination_zone", "")
        zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
        src_zone = command.get("source_zone", "")
        src_str = CardTextResources.get_zone_text(src_zone) if src_zone else ""

        text = text.replace("{value1}", str(amount))
        text = text.replace("{value2}", str(val2))
        text = text.replace("{str_val}", str(str_val))
        text = text.replace("{target}", target_str)
        text = text.replace("{unit}", unit)
        text = text.replace("{zone}", zone_str)
        text = text.replace("{source_zone}", src_str)

        if "{filter}" in text:
            text = text.replace("{filter}", target_str)

        if "{result}" in text:
            from dm_toolkit.gui.i18n import tr
            res = command.get("result", "")
            text = text.replace("{result}", tr(res))
        return text

@register_formatter("MOVE_CARD")
class MoveCardFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        # Convert MOVE_CARD command attributes to TRANSITION style logic for unified formatting
        dest_zone = command.get("destination_zone") or command.get("to_zone", "")
        src_zone = command.get("source_zone", "")

        # Build a temporary command dict resembling a TRANSITION command to reuse TransitionFormatter
        transition_cmd = command.copy()
        transition_cmd["from_zone"] = src_zone
        transition_cmd["to_zone"] = dest_zone

        # Delegate directly to TransitionFormatter to guarantee identical logic
        return TransitionFormatter.format(transition_cmd, ctx)

@register_formatter("REVEAL_TO_BUFFER")
class RevealToBufferFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        src_zone = tr(command.get("from_zone", "DECK"))
        val1 = get_command_amount(command, default=0)
        amt = val1 if val1 > 0 else 1
        text = f"{src_zone}から{amt}枚を表向きにしてバッファに置く。"
        return text

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

            text = f"その中から、{civ_part}{type_part}を{qty_part}選び、{to_zone}に加える。"
            return text
        else:
            qty_part = f"{val1}枚" if val1 > 0 else "すべて"
            if val1 > 0 and up_to:
                qty_part = f"最大{val1}枚"
            text = f"その中から、{qty_part}を{to_zone}に加える。"
            return text

@register_formatter("REPLACE_CARD_MOVE")
class ReplaceCardMoveFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        dest_zone = command.get("destination_zone") or command.get("to_zone", "DECK_BOTTOM")
        src_zone = command.get("source_zone") or command.get("from_zone", "GRAVEYARD")

        zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
        orig_zone_str = CardTextResources.get_zone_text(src_zone) if src_zone else "元のゾーン"
        up_to_flag = bool(command.get('up_to', False))

        scope = command.get("target_group") or command.get("scope", "NONE")
        is_self_ref = scope == "SELF"

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)

        # Base replacement template
        t = CardTextResources.ZONE_MOVE_TEMPLATES.get("REPLACE_CARD_MOVE", "{target}を{from_zone}に置くかわりに{to_zone}に置く。")
        t = t.replace("{from_zone}", orig_zone_str).replace("{to_zone}", zone_str)

        if linked_text:
            input_usage = str(command.get("input_value_usage") or command.get("input_usage") or "").upper()
            linked_target = "そのカード"
            if input_usage == "REPLACEMENT":
                 return f"かわりに、{orig_zone_str}に置くかわりに{zone_str}に置く。"

            t = t.replace("{target}", linked_target)
            return f"その後、{t}"

        amount = get_command_amount(command, default=0)
        target_str, unit = cls._resolve_target(command, ctx.is_spell)

        if is_self_ref:
             t = t.replace("{target}", "このカード")
        else:
             if amount > 0:
                  qty = f"最大{amount}{unit}" if up_to_flag else f"{amount}{unit}"
                  t = f"{target_str}を{qty}選び、" + t.replace("{target}を", "")
             else:
                  t = t.replace("{target}", target_str)

        return t
