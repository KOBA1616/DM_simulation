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
        cond = CardTextGenerator._get_rc_filter_from_effects(card_data)
        if cond and isinstance(cond, dict):
            return f"{kw_str}：{CardTextGenerator._format_revolution_change_text(cond)}"
        return kw_str

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
