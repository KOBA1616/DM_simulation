import re

with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "r") as f:
    content = f.read()

import_stmt1 = "from dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter\n"
import_stmt2 = "from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter\n"
if "QuantityFormatter" not in content:
    content = content.replace("from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter\n",
                              "from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter\n" + import_stmt1 + import_stmt2)

def replace_transition_zones(match):
    return """
        from_z_raw = command.get("from_zone", "")
        if isinstance(from_z_raw, list):
            from_z_list = [CardTextResources.normalize_zone_name(z) for z in from_z_raw]
            from_z = from_z_list[0] if from_z_list else ""
            from_z_str = ZoneFormatter.format_zone_list(from_z_list, context="", joiner="と")
        else:
            from_z = CardTextResources.normalize_zone_name(from_z_raw)
            from_z_str = CardTextResources.get_zone_text(from_z) if from_z else ""

        to_z_raw = command.get("to_zone", "")
        if isinstance(to_z_raw, list):
            to_z_list = [CardTextResources.normalize_zone_name(z) for z in to_z_raw]
            to_z = to_z_list[0] if to_z_list else ""
            to_z_str = ZoneFormatter.format_zone_list(to_z_list, context="", joiner="、または")
        else:
            to_z = CardTextResources.normalize_zone_name(to_z_raw)
            to_z_str = CardTextResources.get_zone_text(to_z) if to_z else ""
"""

content = re.sub(r"        from_z_raw = command\.get\(\"from_zone\", \"\"\).*?to_z_str = CardTextResources\.get_zone_text\(to_z\) if to_z else \"\"", replace_transition_zones, content, flags=re.DOTALL)


def replace_qty_logic(match):
    return """
        # Adjust template for up_to and all
        input_key = command.get('input_value_key') or command.get('input_link')
        is_all = (amount == 0 and not input_key)

        linked_text = None
        if input_key:
             linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)

        formatted_qty = QuantityFormatter.format_quantity(amount, unit, up_to_flag, is_all, linked_text)

        if template_key == ("DECK", "HAND") and target_str == "カード":
             # Exception for generic deck drawing/searching phrasing
             if up_to_flag:
                  qty_str = TextUtils.format_up_to("{amount}", "枚", up_to=True)
                  template = f"山札からカードを{qty_str}選び、手札に加える。"
             else:
                  template = "{from_z}から{target}を{amount}{unit}選び、{to_z}に加える。"
        elif input_key and to_z == "DECK_BOTTOM" and CardTextResources.normalize_zone_name(from_z) == "HAND":
             # Handle input_key dynamic text for deck bottom generic returns from hand
             to_zone_text = CardTextResources.get_zone_text(to_z)
             scope = command.get("target_group") or command.get("scope", "NONE")
             owner = ""
             if scope in ["PLAYER_SELF", "SELF"]:
                 owner = "自分の"
             elif scope in ["PLAYER_OPPONENT", "OPPONENT"]:
                 owner = "相手の"
             elif scope == "ALL_PLAYERS":
                 owner = "各プレイヤーの"

             template = f"{owner}手札から{{target}}を{formatted_qty}選び、{to_zone_text}に置く。"
        elif input_key and to_z == "DECK_BOTTOM":
             template = f"{{from_z}}の{{target}}を{formatted_qty}選び、{{to_z}}に置く。"
        else:
             template = QuantityFormatter.apply_to_template(template, formatted_qty, is_all, up_to_flag, to_z, from_z)
"""

content = re.sub(r"        # Adjust template for up_to and all.*?        if \"\{from_z\}\" in template:",
                 lambda m: replace_qty_logic(m.group(0)) + '        if "{from_z}" in template:',
                 content,
                 flags=re.DOTALL)

def replace_move_buffer(match):
    return """
        has_filter = bool(civs or types or races)
        val1 = get_command_amount(command, default=0)
        up_to = bool(command.get('up_to', False))
        is_all = (val1 == 0)

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

            qty_part = QuantityFormatter.format_quantity(val1, "枚", up_to, is_all)

            text = f"その中から、{civ_part}{type_part}を{qty_part}選び、{to_zone}に加える。"
            return text
        else:
            qty_part = QuantityFormatter.format_quantity(val1, "枚", up_to, is_all)
            text = f"その中から、{qty_part}を{to_zone}に加える。"
            return text
"""

content = re.sub(r"        has_filter = bool\(civs or types or races\).*?return text", replace_move_buffer, content, flags=re.DOTALL, count=1)

with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "w") as f:
    f.write(content)
