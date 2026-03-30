from typing import Dict, Any, List
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.formatters.condition_registry import ConditionFormatterRegistry, register_condition, ConditionFormatterStrategy
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils

@register_condition("MANA_ARMED")
class ManaArmedConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        val = d.get("value", 0)
        civ_raw = d.get("str_val", "")
        civ = tr(civ_raw)
        return f"マナ武装 {val} ({civ})"

@register_condition("SHIELD_COUNT")
class ShieldCountConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        val = d.get("value", 0)
        op = d.get("op", ">=")
        op_text = TextUtils.format_comparison_operator(op, "")
        return f"自分のシールドが{val}つ{op_text}なら"

@register_condition("CIVILIZATION_MATCH")
class CivilizationMatchConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "マナゾーンに同じ文明があれば"

@register_condition("OPPONENT_DRAW_COUNT")
class OpponentDrawCountConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        val = d.get("value", 0)
        return f"相手がカードを{val}枚目以上引いたなら"

@register_condition("COMPARE_STAT")
class CompareStatConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        key = d.get("stat_key", "")
        op = d.get("op", "=")
        val = d.get("value", 0)
        stat_name, unit = CardTextResources.STAT_KEY_MAP.get(key, (key, ""))

        # `TextUtils.format_comparison_operator` yields things like "1以上".
        # We need "1枚以上" (val + unit + suffix)
        op_text = TextUtils.format_comparison_operator(op, f"{val}{unit}")
        # Custom logic for stat comparisons might expect "より少ない", replace "未満" if it's there.
        op_text = op_text.replace("未満", "より少ない")

        return f"自分の{stat_name}が{op_text}なら"

@register_condition("COMPARE_INPUT")
class CompareInputConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        # action is no longer passed as a separate param, we use ctx to pull input references if needed.
        # However, for compatibility we extract it from `d` or `ctx` safely.
        action = d if d.get("input_link") or d.get("input_value_key") else {}
        val = d.get("value", 0)
        op = d.get("op", ">=")

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

        input_key = action.get("input_value_key") or action.get("input_link") or ""

        input_desc = InputLinkFormatter.resolve_linked_value_text(action, context_commands=ctx.current_commands_list if ctx else None)
        if not input_desc:
            input_desc_map = {
                "spell_count": "墓地の呪文の数",
                "card_count": "カードの数",
                "creature_count": "クリーチャーの数",
                "element_count": "エレメントの数"
            }
            input_desc = input_desc_map.get(input_key, InputLinkFormatter.format_input_source_label(action) or "入力値")

        ival = int(val) if isinstance(val, (int, str)) and str(val).isdigit() else val

        # Special logic: for some ops (like >= with an incremented ival), we adapt the value first
        if op == ">=":
             val_str = f"{ival + 1}" if isinstance(ival, int) else f"{val}"
        else:
             val_str = f"{val}"

        op_text = TextUtils.format_comparison_operator(op, val_str, attribute=input_desc, particle="が")

        return f"{op_text}なら"

@register_condition("PLAYED_WITHOUT_MANA_TARGET")
class PlayedWithoutManaTargetConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "指定した対象をコストを支払わずに出していれば"

@register_condition("MANA_CIVILIZATION_COUNT")
class ManaCivilizationCountConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        val = d.get("value", 0)
        op = d.get("op", ">=")
        op_text = "以上" if op == ">=" else "以下" if op == "<=" else "と同じ" if op in ("=", "==") else ""
        return f"自分のマナゾーンにある文明の数が{val}{op_text}なら"

@register_condition("CARDS_MATCHING_FILTER")
class CardsMatchingFilterConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        filter_def = d.get("filter", {})
        count = d.get("count", 1)
        op = d.get("op", ">=")

        from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter

        # Check if the filter specifies ONLY civilization.
        civs = filter_def.get("civilizations", filter_def.get("civilization", []))
        if isinstance(civs, str):
            civs = [civs]

        # Ensure no other filtering keys are used (including zone and owner, which would make it specific cards)
        only_civs = civs and not any(bool(filter_def.get(k)) for k in filter_def if k not in ("civilization", "civilizations"))

        desc = FilterTextFormatter.describe_simple_filter(filter_def)
        zones = filter_def.get("zone", [])

        zone_text = ""
        if "BATTLE_ZONE" in zones:
            zone_text = "バトルゾーンに"
        elif "MANA_ZONE" in zones:
            zone_text = "マナゾーンに"

        if zone_text and desc.startswith(zone_text):
            desc = desc[len(zone_text):]

        if desc.endswith("の"):
            desc = desc[:-1]

        if only_civs:
            civ_names = "・".join([CardTextResources.get_civilization_text(c) for c in civs])
            desc = civ_names + "の文明"
        elif not desc:
            desc = "カード"

        from dm_toolkit.consts import CARD_TYPE_UNIT_MAP

        # 単位決定ロジックの汎化
        unit = "枚"
        card_types = filter_def.get("card_type", [])
        if card_types:
            # 最初の指定タイプに基づく
            unit = CARD_TYPE_UNIT_MAP.get(card_types[0], "枚")
        elif "Demon Command" in desc or "クリーチャー" in desc or "エレメント" in desc:
            unit = "体"

        if desc.endswith("の文明"):
            unit = "つ"

        op_text = TextUtils.format_comparison_operator(op, f"{count}{unit}")
        op_text = op_text.replace("より多い", "より多く")

        verb = "いる" if unit == "体" else "ある"

        return f"{zone_text}{desc}が{op_text}{verb}なら"

@register_condition("OPPONENT_PLAYED_WITHOUT_MANA")
class OpponentPlayedWithoutManaConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "相手がマナゾーンのカードをタップせずに、クリーチャーを出すか呪文を唱えた時"

@register_condition("DURING_YOUR_TURN")
class DuringYourTurnConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "自分のターン中"

    @classmethod
    def get_suffix(cls) -> str:
        return ""

@register_condition("DURING_OPPONENT_TURN")
class DuringOpponentTurnConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "相手のターン中"

    @classmethod
    def get_suffix(cls) -> str:
        return ""

class ConditionFormatter:
    """Formatter for logic and effect conditions."""

    @classmethod
    def format_condition_text(cls, condition: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        """
        Formats a condition dictionary into Japanese text using Strategy Pattern.
        """
        if not condition:
            return ""

        cond_type = condition.get("type", "NONE")

        formatter_cls = ConditionFormatterRegistry.get_formatter(cond_type)
        if formatter_cls:
            text = formatter_cls.format(condition, ctx)
            if text:
                return text + formatter_cls.get_suffix()

        return ""
