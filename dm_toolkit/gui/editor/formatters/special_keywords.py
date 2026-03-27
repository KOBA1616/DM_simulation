from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.keyword_registry import SpecialKeywordFormatterBase, register_special_keyword
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.consts import MAX_COST_VALUE
from dm_toolkit.gui.editor.formatters.utils import is_input_linked

@register_special_keyword("revolution_change")
class RevolutionChangeFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        # Import inside the function to avoid circular imports
        from dm_toolkit.gui.editor.text_generator import CardTextGenerator
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        cond = cls.get_rc_filter_from_effects(card_data)
        if cond and isinstance(cond, dict):
            return f"{kw_str}：{cls.format_revolution_change_text(cond)}"
        return kw_str

    @classmethod
    def format_revolution_change_text(cls, cond: Dict[str, Any]) -> str:
        """Format REVOLUTION_CHANGE condition summary text from filter definition."""
        parts: List[str] = []
        civs = cond.get("civilizations", []) or []
        if civs:
            parts.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))
        min_cost = cond.get("min_cost", 0)
        max_cost = cond.get("max_cost", MAX_COST_VALUE)
        if is_input_linked(min_cost):
            parts.append("コストその数以上")
        elif is_input_linked(max_cost):
            parts.append("コストその数以下")
        else:
            if isinstance(min_cost, int) and isinstance(max_cost, int):
                has_min = min_cost > 0
                has_max = max_cost > 0 and max_cost not in (MAX_COST_VALUE,)
                if has_min and has_max and min_cost != max_cost:
                    parts.append(f"コスト{min_cost}～{max_cost}")
                elif has_min:
                    parts.append(f"コスト{min_cost}以上")
                elif has_max:
                    parts.append(f"コスト{max_cost}以下")
        races = cond.get("races", []) or []
        noun = "/".join(races) if races else "クリーチャー"
        is_evo = cond.get("is_evolution")
        if is_evo is True:
            noun = "進化" + noun
        elif is_evo is False:
            parts.append("進化以外の")
        adjs = "の".join(parts)
        return f"{adjs}の{noun}" if adjs else noun
    @classmethod
    def get_rc_filter_from_effects(cls, data: dict) -> dict:
        """REVOLUTION_CHANGE コマンドの target_filter を効果ノードから探して返す。
        再発防止: 最新仕様では target_filter を単一の正規入力として扱う。"""
        for eff in data.get("effects", []):
            for cmd in (eff.get("commands", []) if isinstance(eff, dict) else []):
                if not isinstance(cmd, dict):
                    continue
                if cls.is_revolution_change_command(cmd):
                    tf = cmd.get("target_filter")
                    if tf and isinstance(tf, dict):
                        return tf
        return {}
    @classmethod
    def is_revolution_change_command(cls, cmd: Dict[str, Any]) -> bool:
        """Return True only for the current REVOLUTION_CHANGE command type."""
        return cmd.get("type") == "REVOLUTION_CHANGE"
@register_special_keyword("friend_burst")
class FriendBurstFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        cond = card_data.get("friend_burst_condition", {})
        if not cond:
            cond = card_data.get("keywords", {}).get("friend_burst_condition", {})
        if cond and isinstance(cond, dict):
            races = cond.get("races", []) or []
            if races:
                return f"{kw_str}：{'/'.join(races)}"
        return kw_str

@register_special_keyword("mekraid")
class MekraidFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        # MEKRAID is separated into the special keywords section.
        # It has no extra formatting here, just the keyword text.
        return CardTextResources.get_keyword_text(keyword_id)
