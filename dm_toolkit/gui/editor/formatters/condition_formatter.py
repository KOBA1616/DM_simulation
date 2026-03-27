from typing import Dict, Any, List
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.i18n import tr

class ConditionFormatter:
    """Formatter for logic and effect conditions."""

    @classmethod
    def format_condition_text(cls, condition: Dict[str, Any], action: Dict[str, Any] = None) -> str:
        """
        Formats a condition dictionary into Japanese text.
        Merges logic from legacy `_format_condition` and `_format_logic_command`.
        `action` context may be provided for commands like COMPARE_INPUT.
        """
        if not condition:
            return ""

        cond_type = condition.get("type", "NONE")

        # Dispatch table
        handlers = {
            "MANA_ARMED": cls._handle_mana_armed,
            "SHIELD_COUNT": cls._handle_shield_count,
            "CIVILIZATION_MATCH": cls._handle_civ_match,
            "OPPONENT_DRAW_COUNT": cls._handle_opponent_draw_count,
            "COMPARE_STAT": cls._handle_compare_stat,
            "COMPARE_INPUT": lambda d: cls._handle_compare_input(d, action or {}),
            "PLAYED_WITHOUT_MANA_TARGET": cls._handle_played_without_mana,
            "MANA_CIVILIZATION_COUNT": cls._handle_mana_civ_count,
            "CARDS_MATCHING_FILTER": cls._handle_cards_matching_filter,
        }

        handler = handlers.get(cond_type)
        if handler:
            try:
                text = handler(condition)
                if text:
                    return text
            except Exception:
                pass

        return ""

    @classmethod
    def _handle_mana_armed(cls, d: Dict[str, Any]) -> str:
        val = d.get("value", 0)
        civ_raw = d.get("str_val", "")
        civ = tr(civ_raw)
        return f"マナ武装 {val} ({civ})"

    @classmethod
    def _handle_shield_count(cls, d: Dict[str, Any]) -> str:
        val = d.get("value", 0)
        op = d.get("op", ">=")
        op_text = "以上" if op == ">=" else "以下" if op == "<=" else ""
        if op == "=" or op == "==":
            op_text = ""
        return f"自分のシールドが{val}つ{op_text}なら"

    @classmethod
    def _handle_civ_match(cls, d: Dict[str, Any]) -> str:
        return "マナゾーンに同じ文明があれば"

    @classmethod
    def _handle_opponent_draw_count(cls, d: Dict[str, Any]) -> str:
        val = d.get("value", 0)
        return f"相手がカードを{val}枚目以上引いたなら"

    @classmethod
    def _handle_compare_stat(cls, d: Dict[str, Any]) -> str:
        key = d.get("stat_key", "")
        op = d.get("op", "=")
        val = d.get("value", 0)
        stat_name, unit = CardTextResources.STAT_KEY_MAP.get(key, (key, ""))
        if op == ">=":
            op_text = f"{val}{unit}以上"
        elif op == "<=":
            op_text = f"{val}{unit}以下"
        elif op == "=" or op == "==":
            op_text = f"{val}{unit}"
        elif op == ">":
            op_text = f"{val}{unit}より多い"
        elif op == "<":
            op_text = f"{val}{unit}より少ない"
        else:
            op_text = f"{val}{unit}"
        return f"自分の{stat_name}が{op_text}なら"

    @classmethod
    def _handle_compare_input(cls, d: Dict[str, Any], action: Dict[str, Any]) -> str:
        val = d.get("value", 0)
        op = d.get("op", ">=")

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

        input_key = action.get("input_value_key") or action.get("input_link") or ""
        input_desc_map = {
            "spell_count": "墓地の呪文の数",
            "card_count": "カードの数",
            "creature_count": "クリーチャーの数",
            "element_count": "エレメントの数"
        }
        input_desc = input_desc_map.get(input_key, InputLinkFormatter.format_input_source_label(action) or "入力値")

        try:
             ival = int(val)
        except Exception:
             ival = val

        if op == ">=":
             try:
                 op_text = f"{ival + 1}以上"
             except Exception:
                 op_text = f"{val}以上"
        elif op == "<=":
             op_text = f"{val}以下"
        elif op == "=" or op == "==":
             op_text = f"{val}"
        elif op == ">":
             op_text = f"{val}より多い"
        elif op == "<":
             op_text = f"{val}より少ない"
        else:
             op_text = f"{val}"
        return f"{input_desc}が{op_text}なら"

    @classmethod
    def _handle_played_without_mana(cls, d: Dict[str, Any]) -> str:
        return "指定した対象をコストを支払わずに出していれば"

    @classmethod
    def _handle_mana_civ_count(cls, d: Dict[str, Any]) -> str:
        val = d.get("value", 0)
        op = d.get("op", ">=")
        op_text = "以上" if op == ">=" else "以下" if op == "<=" else "と同じ" if op == "=" or op == "==" else ""
        return f"自分のマナゾーンにある文明の数が{val}{op_text}なら"

    @classmethod
    def _handle_cards_matching_filter(cls, d: Dict[str, Any]) -> str:
        filter_def = d.get("filter", {})
        count = d.get("count", 1)
        op = d.get("op", ">=")

        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        desc = CardTextGenerator._describe_simple_filter(filter_def)
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

        if desc in ("闇", "光", "火", "水", "自然"):
            desc = desc + "の文明"
        elif desc in ("闇のカード", "光のカード", "火のカード", "水のカード", "自然のカード"):
            desc = desc.replace("のカード", "の文明")

        if "CREATURE" in filter_def.get("card_type", []) or "Demon Command" in desc or "クリーチャー" in desc:
            unit = "体"
        else:
            unit = "枚"

        op_text = ""
        if op == ">=":
            op_text = f"{count}{unit}以上"
        elif op == "<=":
            op_text = f"{count}{unit}以下"
        elif op == "=" or op == "==":
            op_text = f"{count}{unit}"
        elif op == ">":
            op_text = f"{count}{unit}より多く"
        elif op == "<":
            op_text = f"{count}{unit}未満"

        verb = "いる" if unit == "体" else "ある"

        return f"{zone_text}{desc}が{op_text}{verb}なら"
