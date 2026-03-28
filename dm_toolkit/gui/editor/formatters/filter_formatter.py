from typing import Dict, Any, List, Optional
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.target_scope_resolver import TargetScopeResolver
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
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = 999 if unit == "コスト" else 999999

        is_min_linked = InputLinkFormatter.is_input_linked(min_val, usage=min_usage)
        is_max_linked = InputLinkFormatter.is_input_linked(max_val, usage=max_usage)

        if is_min_linked:
            return f"{unit}{linked_token}以上"
        if is_max_linked:
            return f"{unit}{linked_token}以下"

        min_n = min_val if isinstance(min_val, int) else 0
        max_n = max_val if isinstance(max_val, int) else (999 if unit == "コスト" else 999999)

        default_max = 999 if unit == "コスト" else 999999

        if min_n > 0 and max_n < default_max:
            return f"{unit}{min_n}～{max_n}"
        elif min_n > 0:
            return f"{unit}{min_n}以上"
        elif max_n < default_max:
            return f"{unit}{max_n}以下"

        return ""

    @classmethod
    def format_scope_prefix(cls, scope: str, text: str = "") -> str:
        """
        Applies a scope prefix (e.g., "自分の", "相手の") to a text,
        avoiding duplication like "相手の相手の".
        """
        if not scope or scope == "NONE" or scope == "ALL":
            return text

        scope_text = TargetScopeResolver.resolve_prefix(scope)
        if not scope_text:
            return text

        noun = TargetScopeResolver.resolve_noun(scope)

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
        civs = filter_def.get("civilizations", [])
        races = filter_def.get("races", [])
        types = filter_def.get("types", [])
        min_cost = filter_def.get("min_cost", 0)
        max_cost = filter_def.get("max_cost", consts.MAX_COST_VALUE)
        exact_cost = filter_def.get("exact_cost")
        cost_ref = filter_def.get("cost_ref")

        adjectives = []
        if civs:
            adjectives.append("/".join([CardTextResources.get_civilization_text(c) for c in civs]))

        # Handle cost filtering
        if cost_ref:
            adjectives.append("選択した数字と同じコスト")
        elif exact_cost is not None:
            adjectives.append(f"コスト{exact_cost}")
        else:
            cost_text = FilterTextFormatter.format_range_text(min_cost, max_cost, unit="コスト", linked_token="その数")
            if cost_text:
                adjectives.append(cost_text)

        adj_str = "の".join(adjectives)
        if adj_str:
            adj_str += "の"

        # 再発防止: types が空のときに「クリーチャー」をデフォルトにしない。
        #   フィルターでタイプ未指定は「カード」(全タイプ対象)。
        #   CREATURE のみ指定時だけ「クリーチャー」、SPELL のみなら「呪文」、
        #   複数タイプ指定時は "/" 区切り、races 指定があればそれを優先する。
        if consts.CardType.ELEMENT.value in types:
            noun_str = "エレメント"
        elif consts.CardType.SPELL.value in types and consts.CardType.CREATURE.value not in types:
            noun_str = "呪文"
        elif consts.CardType.CREATURE.value in types:
            noun_str = "クリーチャー"
        elif types:
            noun_str = "/".join(tr(t) for t in types if t)
        else:
            noun_str = "カード"  # タイプ未指定は全タイプ対象

        if races:
            noun_str = "/".join(races)

        return adj_str + noun_str
