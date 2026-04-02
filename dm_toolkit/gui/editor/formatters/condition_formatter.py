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
        template = "マナ武装 {val} ({civ})"

        # Proper Python template engine approach using format() instead of procedural loops
        # or error-prone replace() chaining.
        return template.format(val=val, civ=civ)

@register_condition("SHIELD_COUNT")
class ShieldCountConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        val = d.get("value", 0)
        op = d.get("op", ">=")
        op_text = TextUtils.format_comparison_operator(op, "")
        return f"自分のシールドが{val}つ{op_text}"

@register_condition("CIVILIZATION_MATCH")
class CivilizationMatchConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "マナゾーンに同じ文明があれば"

    @classmethod
    def get_suffix(cls) -> str:
        return "、"

@register_condition("OPPONENT_DRAW_COUNT")
class OpponentDrawCountConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        val = d.get("value", 0)
        return f"{val}枚目以降なら"

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

        prefix = "自分の" if not stat_name.startswith("自分の") else ""
        return f"{prefix}{stat_name}が{op_text}"

@register_condition("COMPARE_INPUT")
class CompareInputConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        # action is no longer passed as a separate param, we use ctx to pull input references if needed.
        # However, for compatibility we extract it from `d` or `ctx` safely.
        # Some callers might pass action dict as ctx due to old APIs, so we handle both.
        if isinstance(ctx, dict):
             action = ctx
             current_cmds = None
        else:
             action = d if d.get("input_link") or d.get("input_value_key") else {}
             current_cmds = ctx.current_commands_list if ctx else None

        val = d.get("value", 0)
        op = d.get("op", ">=")

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        from dm_toolkit.gui.editor.formatters.input_link_ast import InputLinkASTBuilder

        input_key = action.get("input_value_key") or action.get("input_link") or ""

        # Delegate deeper AST resolution specifically for COMPARISON sentences where natural wording differs
        input_desc = None
        if current_cmds and input_key:
            producer = InputLinkASTBuilder.find_producer(current_cmds, input_key)
            if producer:
                if producer.atype == "DRAW_CARD":
                    input_desc = "引いた枚数"
                elif producer.atype == "DISCARD":
                    input_desc = "捨てた枚数"
                elif producer.atype in ("DECLARE_NUMBER", "SELECT_NUMBER"):
                    input_desc = "選択した数"

        if not input_desc:
             input_desc = InputLinkFormatter.resolve_linked_value_text(action, context_commands=current_cmds)
             # Strip leading "その" and trailing "と同じ" for cleaner phrasing in a conditional clause
             if input_desc.startswith("その"):
                 input_desc = input_desc[2:]
             if input_desc.endswith("と同じ"):
                 input_desc = input_desc[:-3]
             elif input_desc.endswith("と同じ数"):
                 input_desc = input_desc[:-4] + "数"

        if not input_desc:
            input_desc_map = {
                "spell_count": "墓地の呪文の数",
                "card_count": "カードの数",
                "creature_count": "クリーチャーの数",
                "element_count": "エレメントの数"
            }
            input_desc = input_desc_map.get(input_key, InputLinkFormatter.format_input_source_label(action) or "入力値")

        ival = int(val) if isinstance(val, (int, str)) and str(val).isdigit() else val

        # Adjust formatting unit based on semantic context
        val_suffix = ""
        if "枚数" in input_desc:
             val_suffix = "枚"

        # Special logic: for some ops (like >= with an incremented ival), we adapt the value first
        if op == ">=":
             # If op is >= and we are comparing count/枚数, "X枚以上" natively handled
             val_str = f"{ival}{val_suffix}"
             # For legacy reasons some cards expect strictly greater semantics, but >= is standard "以上"
        else:
             val_str = f"{val}{val_suffix}"

        op_text = TextUtils.format_comparison_operator(op, val_str, attribute=input_desc, particle="が")

        return f"{op_text}"

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
        return f"自分のマナゾーンにある文明の数が{val}{op_text}"

@register_condition("CARDS_MATCHING_FILTER")
class CardsMatchingFilterConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        filter_def = d.get("filter", {})
        if not filter_def and "target_filter" in d:
             filter_def = d.get("target_filter", {})

        count = d.get("count", 1)
        op = d.get("op", ">=")

        from dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService

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

        if not desc:
            desc = "カード"

        # 単位決定ロジックの汎化
        from dm_toolkit.consts import CARD_TYPE_UNIT_MAP
        card_types = filter_def.get("card_type", filter_def.get("types", []))

        # TargetResolutionService already computes optimal unit for types/zones
        _, _, resolved_unit = TargetResolutionService._determine_noun(zones, card_types, "")
        unit = resolved_unit

        # Override for purely civilization checks (no types specified, outputs "文明")
        if desc.endswith("の文明"):
            unit = "枚" # Many tests explicitly expect 枚 here based on past formatting

        op_text = TextUtils.format_comparison_operator(op, f"{count}{unit}")
        op_text = op_text.replace("より多い", "より多く")

        verb = "いる" if unit == "体" else "ある"

        return f"{zone_text}{desc}が{op_text}{verb}"

@register_condition("OPPONENT_PLAYED_WITHOUT_MANA")
class OpponentPlayedWithoutManaConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "相手がマナゾーンのカードをタップせずに、クリーチャーを出すか呪文を唱えた"

    @classmethod
    def get_suffix(cls) -> str:
        return "時、"

@register_condition("DURING_YOUR_TURN")
class DuringYourTurnConditionFormatter(ConditionFormatterStrategy):
    is_active_condition_prefix = True

    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "自分のターン"

    @classmethod
    def get_suffix(cls) -> str:
        return "中、"

@register_condition("DURING_OPPONENT_TURN")
class DuringOpponentTurnConditionFormatter(ConditionFormatterStrategy):
    is_active_condition_prefix = True

    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        return "相手のターン"

    @classmethod
    def get_suffix(cls) -> str:
        return "中、"

class ConditionFormatter:
    """Formatter for logic and effect conditions."""

    @classmethod
    def format_condition_text(cls, condition: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        """
        Formats a condition dictionary into Japanese text using Strategy Pattern.
        """
        if not condition:
            return ""

        from dm_toolkit.gui.editor.formatters.clause_joiner import ClauseJoiner
        return ClauseJoiner.join_condition_ast(condition, ctx)
