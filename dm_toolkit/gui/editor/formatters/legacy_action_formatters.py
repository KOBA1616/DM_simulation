from typing import Dict, Any
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount, is_input_linked
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.consts import MAX_COST_VALUE

class LegacyActionFormatterHelper:
    @staticmethod
    def get_base_template(atype: str) -> str:
        return CardTextResources.ACTION_MAP.get(atype, "")

    @staticmethod
    def handle_input_link(atype: str, command: Dict[str, Any], template: str) -> tuple[str, str]:
        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command)
        input_usage = command.get("input_value_usage") or command.get("input_usage")
        val1 = get_command_amount(command, default=0)

        if not linked_text:
            return template, str(val1)

        usage_label_suffix = ""
        if input_usage:
            label = InputLinkFormatter.format_input_usage_label(input_usage)
            if label:
                usage_label_suffix = f"（{label}）"

        up_to_flag = bool(command.get('up_to', False))

        if atype == "DESTROY":
            if up_to_flag:
                template = f"{{target}}をその同じ数だけまで選び、破壊する。{usage_label_suffix}"
            else:
                template = f"{{target}}をその同じ数だけ破壊する。{usage_label_suffix}"
        elif atype == "TAP":
            if up_to_flag:
                template = f"{{target}}をその同じ数だけまで選び、タップする。{usage_label_suffix}"
            else:
                template = f"{{target}}をその同じ数だけ選び、タップする。{usage_label_suffix}"
        elif atype == "UNTAP":
            if up_to_flag:
                template = f"{{target}}をその同じ数だけまで選び、アンタップする。{usage_label_suffix}"
            else:
                template = f"{{target}}をその同じ数だけ選び、アンタップする。{usage_label_suffix}"
        elif atype == "RETURN_TO_HAND":
            if up_to_flag:
                template = f"{{target}}をその同じ数だけまで選び、手札に戻す。{usage_label_suffix}"
            else:
                template = f"{{target}}をその同じ数だけ選び、手札に戻す。{usage_label_suffix}"
        elif atype == "SEND_TO_MANA":
            if up_to_flag:
                template = f"{{target}}をその同じ数だけまで選び、マナゾーンに置く。{usage_label_suffix}"
            else:
                template = f"{{target}}をその同じ数だけ選び、マナゾーンに置く。{usage_label_suffix}"
        else:
            val1 = "その数"

        return template, str(val1)

    @staticmethod
    def handle_all_selection(atype: str, command: Dict[str, Any], template: str, val1: Any) -> str:
        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command)

        if str(val1) == "0" and not linked_text:
            if atype == "DESTROY": return "{target}をすべて破壊する。"
            if atype == "TAP": return "{target}をすべてタップする。"
            if atype == "UNTAP": return "{target}をすべてアンタップする。"
            if atype == "RETURN_TO_HAND": return "{target}をすべて手札に戻す。"
            if atype == "SEND_TO_MANA": return "{target}をすべてマナゾーンに置く。"

        return template

    @staticmethod
    def apply_replacements(command: Dict[str, Any], ctx: TextGenerationContext, template: str, val1: str, target_str: str, unit: str) -> str:
        val2 = command.get("value2", 0)
        str_val = command.get("str_param") or command.get("str_val", "")

        dest_zone = command.get("destination_zone", "")
        zone_str = CardTextResources.get_zone_text(dest_zone) if dest_zone else "どこか"
        src_zone = command.get("source_zone", "")
        src_str = CardTextResources.get_zone_text(src_zone) if src_zone else ""

        text = template.replace("{value1}", str(val1))
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



class BaseGenericLegacyFormatter(CommandFormatterBase):
    atype = ""
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        template = LegacyActionFormatterHelper.get_base_template(cls.atype)
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        template, val1_str = LegacyActionFormatterHelper.handle_input_link(cls.atype, command, template)
        template = LegacyActionFormatterHelper.handle_all_selection(cls.atype, command, template, val1_str)

        # Override template for DESTROY if targeting trigger source
        if cls.atype == "DESTROY" and command.get('filter', {}).get('is_trigger_source'):
            template = "{target}を破壊する。"

        text = LegacyActionFormatterHelper.apply_replacements(command, ctx, template, val1_str, target_str, unit)
        return text

@register_formatter("DESTROY")
class DestroyFormatter(BaseGenericLegacyFormatter):
    atype = "DESTROY"

@register_formatter("TAP")
class TapFormatter(BaseGenericLegacyFormatter):
    atype = "TAP"

@register_formatter("UNTAP")
class UntapFormatter(BaseGenericLegacyFormatter):
    atype = "UNTAP"

@register_formatter("RETURN_TO_HAND")
class ReturnToHandFormatter(BaseGenericLegacyFormatter):
    atype = "RETURN_TO_HAND"

@register_formatter("SEND_TO_MANA")
class SendToManaFormatter(BaseGenericLegacyFormatter):
    atype = "SEND_TO_MANA"

@register_formatter("COST_REDUCTION")
class CostReductionFormatter(BaseGenericLegacyFormatter):
    atype = "COST_REDUCTION"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        text = super().format(command, ctx)
        target_str, _ = cls._resolve_target(command, ctx.is_spell)

        if target_str == "カード" or target_str == "自分のカード":
            replacement = "この呪文" if ctx.is_spell else "このクリーチャー"
            text = text.replace("カード", replacement)
            text = text.replace("自分のカード", replacement)

        cond = command.get("condition", {})
        if cond:
            cond_text = CardTextGenerator._format_condition(cond)
            text = f"{cond_text}{text}"

        return text

@register_formatter("GRANT_KEYWORD")
class GrantKeywordFormatter(BaseGenericLegacyFormatter):
    atype = "GRANT_KEYWORD"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        str_val = command.get("str_param") or command.get("str_val", "")
        keyword = CardTextResources.get_keyword_text(str_val)
        command_copy = command.copy()
        command_copy["str_val"] = keyword
        return super().format(command_copy, ctx)

@register_formatter("PLAY_FROM_ZONE")
class PlayFromZoneFormatter(CommandFormatterBase):
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        action = command.copy()
        temp_filter = action.get("filter", {}).copy()
        action["filter"] = temp_filter

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(action)
        input_usage = action.get("input_value_usage") or action.get("input_usage")
        usage_label_suffix = ""
        if linked_text and input_usage:
            label = InputLinkFormatter.format_input_usage_label(input_usage)
            if label:
                usage_label_suffix = f"（{label}）"

        if not action.get("source_zone") and "zones" in temp_filter:
            zones = temp_filter["zones"]
            if len(zones) == 1:
                action["source_zone"] = zones[0]

        if action.get("value1", 0) == 0:
            max_cost = command.get("max_cost")
            if max_cost is None:
                max_cost = temp_filter.get("max_cost", MAX_COST_VALUE)
            if is_input_linked(max_cost):
                pass
            elif max_cost < MAX_COST_VALUE:
                action["value1"] = max_cost
                if "max_cost" in temp_filter: del temp_filter["max_cost"]
        else:
             if "max_cost" in temp_filter:
                  del temp_filter["max_cost"]

        if "zones" in temp_filter: temp_filter["zones"] = []
        scope = action.get("target_group") or action.get("scope", "NONE")
        if scope in ["PLAYER_SELF", "SELF"]: action["scope"] = "NONE"

        use_linked_cost = False
        max_cost = temp_filter.get("max_cost")
        if is_input_linked(max_cost, usage="MAX_COST"):
            use_linked_cost = True

        omit_cost = False
        if use_linked_cost and action.get("value1", 0) == 0:
            omit_cost = True

        target_str, unit = cls._resolve_target(action, ctx.is_spell, omit_cost=omit_cost)
        count = temp_filter.get("count")
        count_text = ""
        if isinstance(count, int) and count > 0:
            count_text = f"{count}{unit}選び、"

        verb = "プレイする"
        types = temp_filter.get("types", [])
        if "SPELL" in types and "CREATURE" not in types:
            verb = "唱える"
        elif "CREATURE" in types:
            verb = "召喚する"

        play_flags = action.get("play_flags")
        is_free = False
        if isinstance(play_flags, bool) and play_flags:
            is_free = True
        elif isinstance(play_flags, list) and ("FREE" in play_flags or "COST_FREE" in play_flags):
            is_free = True

        if is_free:
            verb = f"コストを支払わずに{verb}"

        linked_cost_phrase = ""
        if use_linked_cost:
            source_token = InputLinkFormatter.format_linked_count_token(action, "その数")
            source_token = InputLinkFormatter.normalize_linked_count_label(source_token)
            linked_cost_phrase = f"{source_token}以下のコストの"

        if use_linked_cost:
            if not omit_cost and target_str.startswith("コストその数以下の"):
                target_str = target_str.replace("コストその数以下の", "", 1)
            if linked_cost_phrase and not target_str.startswith(linked_cost_phrase):
                target_str = linked_cost_phrase + target_str

            if action.get("source_zone"):
                template = "{source_zone}から{target}を" + count_text + verb + f"。{usage_label_suffix}"
            else:
                template = "{target}を" + count_text + verb + f"。{usage_label_suffix}"
        else:
            if action.get("value1", 0) == 0:
                # If value1 is 0 and not linked cost, fallback to just "{target}を..."
                if action.get("source_zone"):
                    template = "{source_zone}から{target}を" + count_text + verb + f"。{usage_label_suffix}"
                else:
                    template = "{target}を" + count_text + verb + f"。{usage_label_suffix}"
            else:
                if action.get("source_zone"):
                    template = "{source_zone}からコスト{value1}以下の{target}を" + count_text + verb + f"。{usage_label_suffix}"
                else:
                    template = "コスト{value1}以下の{target}を" + count_text + verb + f"。{usage_label_suffix}"

        text = LegacyActionFormatterHelper.apply_replacements(action, ctx, template, str(action.get("value1", 0)), target_str, unit)

        return text
