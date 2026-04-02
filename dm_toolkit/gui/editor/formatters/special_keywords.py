from typing import Dict, Any, List
from dm_toolkit.gui.editor.formatters.keyword_registry import SpecialKeywordFormatterBase, register_special_keyword
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.consts import MAX_COST_VALUE
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

@register_special_keyword("revolution_change")
class RevolutionChangeFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        cond = cls.extract_requirements(card_data)
        if cond and isinstance(cond, dict):
            return f"{kw_str}：{cls.format_revolution_change_text(cond)}"
        return kw_str

    @classmethod
    def get_unbound_text(cls, card_data: Dict[str, Any]) -> List[str]:
        lines = []
        kw_str = CardTextResources.get_keyword_text("revolution_change")
        cond = cls.extract_requirements(card_data)
        if cond and isinstance(cond, dict):
            lines.append(f"{kw_str}：{cls.format_revolution_change_text(cond)}")
        return lines

    @classmethod
    def format_revolution_change_text(cls, cond: Dict[str, Any]) -> str:
        """Format REVOLUTION_CHANGE condition summary text from filter definition."""
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService

        # We need to omit races here to avoid repetition since we add them at the end
        import copy
        cond_no_race = copy.deepcopy(cond)
        if "races" in cond_no_race:
            del cond_no_race["races"]

        parts: List[str] = TargetResolutionService.build_attribute_list(cond_no_race)

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
    def extract_requirements(cls, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract revolution change source condition explicitly from card_data, falling back to scanning effects."""
        cond = card_data.get("revolution_change_condition", {})
        if not cond:
            cond = card_data.get("keywords", {}).get("revolution_change_condition", {})
        if cond and isinstance(cond, dict):
            return cond

        # Fallback to scanning if declarative condition is missing (for legacy compatibility)
        return cls.get_rc_filter_from_effects(card_data)

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
        return cls.format_numbered_keyword(keyword_id, card_data)

@register_special_keyword("mega_last_burst")
class MegaLastBurstFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        return kw_str + "（このクリーチャーが手札、マナゾーン、または墓地に置かれた時、このカードの呪文側をコストを支払わずに唱えてもよい）"

@register_special_keyword("power_attacker")
class PowerAttackerFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        bonus = card_data.get("power_attacker_bonus", 0)
        if bonus > 0:
            kw_str += f" +{bonus}"
        return kw_str

@register_special_keyword("hyper_energy")
class HyperEnergyFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        return kw_str + "（このクリーチャーを召喚する時、コストが異なる自分のクリーチャーを好きな数タップしてもよい、こうしてタップしたクリーチャー1体につき、このクリーチャーの召喚コストを2少なくする、ただし、コストは0以下にならない。）"

@register_special_keyword("just_diver")
class JustDiverFormatter(SpecialKeywordFormatterBase):
    @classmethod
    def format(cls, keyword_id: str, card_data: Dict[str, Any]) -> str:
        kw_str = CardTextResources.get_keyword_text(keyword_id)
        return kw_str + "（このクリーチャーが出た時、次の自分のターンのはじめまで、このクリーチャーは相手に選ばれず、攻撃されない）"
