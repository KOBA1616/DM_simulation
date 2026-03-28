from typing import Dict, Any
from dm_toolkit.gui.editor.formatters.command_formatter_base import CommandFormatterBase
from dm_toolkit.gui.editor.formatters.command_registry import register_formatter
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.utils import get_command_amount, is_input_linked
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.consts import MAX_COST_VALUE


class BaseGenericLegacyFormatter(CommandFormatterBase):
    atype = ""
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        # Default fallback - should be overridden by child classes
        return ""

    @classmethod
    def _get_val_str(cls, command: Dict[str, Any]) -> str:
        # helper to get val str and usage label
        linked_text = InputLinkFormatter.resolve_linked_value_text(command)
        if linked_text:
            return "その数"
        return str(get_command_amount(command, default=0))

    @classmethod
    def _get_usage_label(cls, command: Dict[str, Any]) -> str:
        linked_text = InputLinkFormatter.resolve_linked_value_text(command)
        if not linked_text:
            return ""
        input_usage = command.get("input_value_usage") or command.get("input_usage")
        if input_usage:
            label = InputLinkFormatter.format_input_usage_label(input_usage)
            if label:
                return f"（{label}）"
        return ""


@register_formatter("DESTROY")
class DestroyFormatter(BaseGenericLegacyFormatter):
    atype = "DESTROY"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val_str = cls._get_val_str(command)
        usage_label = cls._get_usage_label(command)

        if command.get('filter', {}).get('is_trigger_source'):
            return f"{target_str}を破壊する。"

        if val_str == "0":
            return f"{target_str}をすべて破壊する。"

        if val_str == "その数":
            link_phrase = InputLinkFormatter.format_input_link_text(command, cls.atype)
            return f"{target_str}を{link_phrase}破壊する。{usage_label}"

        return f"{target_str}を{val_str}{unit}破壊する。{usage_label}"

@register_formatter("TAP")
class TapFormatter(BaseGenericLegacyFormatter):
    atype = "TAP"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val_str = cls._get_val_str(command)
        usage_label = cls._get_usage_label(command)

        if val_str == "0":
            return f"{target_str}をすべてタップする。"

        if val_str == "その数":
            link_phrase = InputLinkFormatter.format_input_link_text(command, cls.atype)
            return f"{target_str}を{link_phrase}タップする。{usage_label}"

        return f"{target_str}を{val_str}{unit}選び、タップする。{usage_label}"

@register_formatter("UNTAP")
class UntapFormatter(BaseGenericLegacyFormatter):
    atype = "UNTAP"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val_str = cls._get_val_str(command)
        usage_label = cls._get_usage_label(command)

        if val_str == "0":
            return f"{target_str}をすべてアンタップする。"

        if val_str == "その数":
            link_phrase = InputLinkFormatter.format_input_link_text(command, cls.atype)
            return f"{target_str}を{link_phrase}アンタップする。{usage_label}"

        return f"{target_str}を{val_str}{unit}選び、アンタップする。{usage_label}"

@register_formatter("RETURN_TO_HAND")
class ReturnToHandFormatter(BaseGenericLegacyFormatter):
    atype = "RETURN_TO_HAND"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val_str = cls._get_val_str(command)
        usage_label = cls._get_usage_label(command)

        if val_str == "0":
            return f"{target_str}をすべて手札に戻す。"

        if val_str == "その数":
            link_phrase = InputLinkFormatter.format_input_link_text(command, cls.atype)
            return f"{target_str}を{link_phrase}手札に戻す。{usage_label}"

        return f"{target_str}を{val_str}{unit}選び、手札に戻す。{usage_label}"

@register_formatter("SEND_TO_MANA")
class SendToManaFormatter(BaseGenericLegacyFormatter):
    atype = "SEND_TO_MANA"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        val_str = cls._get_val_str(command)
        usage_label = cls._get_usage_label(command)

        if val_str == "0":
            return f"{target_str}をすべてマナゾーンに置く。"

        if val_str == "その数":
            link_phrase = InputLinkFormatter.format_input_link_text(command, cls.atype)
            return f"{target_str}を{link_phrase}マナゾーンに置く。{usage_label}"

        return f"{target_str}を{val_str}{unit}選び、マナゾーンに置く。{usage_label}"

@register_formatter("COST_REDUCTION")
class CostReductionFormatter(BaseGenericLegacyFormatter):
    atype = "COST_REDUCTION"
    @classmethod
    def format(cls, command: Dict[str, Any], ctx: TextGenerationContext) -> str:
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator

        default_noun = "この呪文" if ctx.is_spell else "このクリーチャー"
        target_str, unit = cls._resolve_target(command, ctx.is_spell, default_self_noun=default_noun)
        val_str = cls._get_val_str(command)
        usage_label = cls._get_usage_label(command)

        text = f"{target_str}のコストを{val_str}少なくする。ただし、コストは0以下にはならない。"

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
        target_str, unit = cls._resolve_target(command, ctx.is_spell)
        return f"{target_str}に「{keyword}」を与える。"

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

        val1 = action.get("value1", 0)
        source_zone = action.get("source_zone")
        zone_str = CardTextResources.get_zone_text(source_zone) if source_zone else ""

        if use_linked_cost:
            if source_zone:
                return f"{zone_str}から{target_str}を{count_text}{verb}。{usage_label_suffix}"
            else:
                return f"{target_str}を{count_text}{verb}。{usage_label_suffix}"
        else:
            if val1 == 0:
                if source_zone:
                    return f"{zone_str}から{target_str}を{count_text}{verb}。{usage_label_suffix}"
                else:
                    return f"{target_str}を{count_text}{verb}。{usage_label_suffix}"
            else:
                if source_zone:
                    return f"{zone_str}からコスト{val1}以下の{target_str}を{count_text}{verb}。{usage_label_suffix}"
                else:
                    return f"コスト{val1}以下の{target_str}を{count_text}{verb}。{usage_label_suffix}"

