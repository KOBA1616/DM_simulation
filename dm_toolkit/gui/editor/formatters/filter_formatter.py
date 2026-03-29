from typing import Dict, Any, List, Optional
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.value_resolver import ValueResolver
from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
from dm_toolkit import consts

class FilterTextFormatter:
    """
    Centralized formatting for target and filter descriptions.
    """
    @classmethod
    def format_range_text(cls, min_val: Any, max_val: Any, unit: str = "コスト", min_usage: str = "MIN_COST", max_usage: str = "MAX_COST", linked_token: str = "その数") -> str:
        """
        Formats a range description (e.g. "コスト3～5", "パワー5000以上").
        Returns the formatted string without trailing particles like "の".
        """
        return ValueResolver.resolve_range(
            min_val=min_val, max_val=max_val, unit=unit,
            min_usage=min_usage, max_usage=max_usage, linked_token=linked_token
        )

    @classmethod
    def format_scope_prefix(cls, scope: str, text: str = "") -> str:
        """
        Applies a scope prefix (e.g., "自分の", "相手の") to a text,
        avoiding duplication like "相手の相手の".
        """
        if not scope or scope == "NONE" or scope == "ALL":
            return text

        scope_text = TargetResolutionService.resolve_prefix(scope)
        if not scope_text:
            return text

        noun = TargetResolutionService.resolve_noun(scope)

        scope_variants = []
        if noun:
            scope_variants.extend([f"{noun}が", f"{noun}の"])

        for v in scope_variants:
            if v in text:
                return text

        # Handle cases where we have just "自分" or "相手"
        if not text:
            return scope_text

        if scope_text.endswith("の") and text.startswith("の"):
            return scope_text + text[1:]

        return f"{scope_text}{text}"

    @classmethod
    def describe_simple_filter(cls, filter_def: Dict[str, Any]) -> str:
        types = filter_def.get("types", [])
        races = filter_def.get("races", [])

        adjectives = TargetResolutionService.build_attribute_list(filter_def)

        adj_str = "の".join(adjectives)
        if adj_str:
            adj_str += "の"

        # 再発防止: types が空のときに「クリーチャー」をデフォルトにしない。
        #   フィルターでタイプ未指定は「カード」(全タイプ対象)。
        #   CREATURE のみ指定時だけ「クリーチャー」、SPELL のみなら「呪文」、
        #   複数タイプ指定時は "/" 区切り、races 指定があればそれを優先する。
        if consts.CardTypeHelper.is_element_like(types) and (consts.CardType.ELEMENT.value in types or not consts.CardTypeHelper.is_creature_like(types)) and not consts.CardTypeHelper.is_spell_like(types):
            noun_str = "エレメント"
        elif consts.CardTypeHelper.is_spell_like(types) and not consts.CardTypeHelper.is_creature_like(types):
            noun_str = "呪文"
        elif consts.CardTypeHelper.is_creature_like(types):
            noun_str = "クリーチャー"
        elif types:
            noun_str = "/".join(tr(t) for t in types if t)
        else:
            noun_str = "カード"  # タイプ未指定は全タイプ対象

        if races:
            noun_str = "/".join(races)

        return adj_str + noun_str
