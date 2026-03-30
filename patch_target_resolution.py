import re

with open("dm_toolkit/gui/editor/services/target_resolution_service.py", "r") as f:
    content = f.read()

import_stmt = "from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter\n"
if "from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter" not in content:
    content = content.replace("from dm_toolkit.gui.editor.text_resources import CardTextResources\n",
                              "from dm_toolkit.gui.editor.text_resources import CardTextResources\n" + import_stmt)

def replace_target_zones(match):
    return """
        if is_modifier:
            if zones:
                if len(zones) == 1:
                     parts.append(ZoneFormatter.format_zone_list(zones, context="in"))
                else:
                     parts.append(ZoneFormatter.format_zone_list(zones, context="from", joiner="または"))
        elif is_trigger:
            if zones and Zone.BATTLE_ZONE.value not in zones:
                parts.append(ZoneFormatter.format_zone_list(zones, context="in", joiner="、または"))
        else:
            if zone_noun: parts.append(zone_noun + "の")
"""

content = re.sub(r"        if is_modifier:.*?            if zone_noun: parts\.append\(zone_noun \+ \"の\"\)", replace_target_zones, content, flags=re.DOTALL)


def replace_format_modifier(match):
    return """    def format_modifier_target(cls, filter_def: Dict[str, Any], scope: str = "ALL") -> str:
        \"\"\"Format target description from filter with comprehensive support, including scope.\"\"\"
        if not filter_def and scope == "NONE":
            return "このクリーチャー"
        elif not filter_def:
             return "対象"

        effective_filter = filter_def.copy() if filter_def else {}
        if scope and scope != "ALL":
             effective_filter["owner"] = scope

        owner = effective_filter.get("owner", "NONE")
        if owner == "NONE" and not effective_filter.get("zones") and not effective_filter.get("types") and not effective_filter.get("races") and not effective_filter.get("civilizations"):
            return "このクリーチャー"

        result, _ = cls.build_subject(effective_filter, is_modifier=True)

        from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
        effective_scope = cls.resolve_effective_owner(scope, effective_filter)
        return FilterTextFormatter.format_scope_prefix(effective_scope, result)"""

content = re.sub(r"    @classmethod\n    def format_modifier_target.*?return result", "@classmethod\n" + replace_format_modifier(None), content, flags=re.DOTALL)


def replace_format_target(match):
    return """        prefix, effective_scope = cls._resolve_scope(scope, filter_def)

        if not filter_def and scope == "NONE":
             # This means target self implicitly
             target_desc = default_self_noun if default_self_noun else "このカード"
             return target_desc, unit

        if filter_def:
            input_usage = action.get("input_value_usage") or action.get("input_usage")
            has_input_key = bool(action.get("input_value_key") or action.get("input_link"))
            target_desc, unit = cls.build_subject(filter_def, omit_cost=omit_cost, input_usage=input_usage, has_input_key=has_input_key, action_type=atype)

            from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter

            # Use FilterTextFormatter.format_scope_prefix for standard handling
            # If prefix was custom (like ランダムな), apply it directly instead
            if prefix and prefix not in ["すべてのプレイヤーの", "すべてのプレイヤー"]:
                target_desc = prefix + target_desc
            else:
                target_desc = FilterTextFormatter.format_scope_prefix(effective_scope, target_desc)

            zones = filter_def.get("zones", [])
            types = filter_def.get("types", [])
            if Zone.SHIELD_ZONE.value in zones and (not types or CardType.CARD.value in types):
                target_desc = target_desc.replace("シールドゾーンのカード", "シールド")
                unit = "つ"

        else:
            target_desc = ""
            unit = "枚"
"""

content = re.sub(r"        prefix, effective_scope = cls\._resolve_scope\(scope, filter_def\).*?unit = \"枚\"", replace_format_target, content, flags=re.DOTALL)

with open("dm_toolkit/gui/editor/services/target_resolution_service.py", "w") as f:
    f.write(content)
