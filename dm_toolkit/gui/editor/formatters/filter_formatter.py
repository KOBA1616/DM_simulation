from typing import Dict, Any, List, Optional
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
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
        Applies a scope prefix (e.g., "自分の", "相手の") to a text based on explicit data structure.
        """
        if not scope or scope == "NONE" or scope == "ALL":
            return text

        scope_text = TargetResolutionService.resolve_prefix(scope)
        if not scope_text:
            return text

        # If text is empty, return just the prefix.
        if not text:
            return scope_text

        # Append correctly, avoiding double particles.
        if scope_text.endswith("の") and text.startswith("の"):
            return scope_text + text[1:]

        # Basic check to avoid gross duplication if somehow it got through
        noun = TargetResolutionService.resolve_noun(scope)
        if noun and (text.startswith(f"{noun}の") or text.startswith(f"{noun}が")):
            return text

        return f"{scope_text}{text}"

    @classmethod
    def describe_simple_filter(cls, filter_def: Dict[str, Any]) -> str:
        types = filter_def.get("types", [])
        races = filter_def.get("races", [])

        adjectives = TargetResolutionService.build_attribute_list(filter_def)

        # Races shouldn't be in both adjectives and noun string to avoid duplication like "Demon CommandのDemon Command"
        if races:
            adj_races_str = "/".join(races)
            if adj_races_str in adjectives:
                adjectives.remove(adj_races_str)

        adj_str = "の".join(adjectives)
        if adj_str:
            adj_str += "の"

        # Check if the filter specifies ONLY civilization.
        civs = filter_def.get("civilizations", filter_def.get("civilization", []))
        if isinstance(civs, str):
            civs = [civs]

        # Ensure no other filtering keys are used (including zone and owner, which would make it specific cards)
        # Note: 'zone' is used in conditions, 'zones' in proper filters. Ignore them here.
        ignored_keys = {"civilization", "civilizations", "zone", "zones", "owner", "types", "exclude"}
        has_other_keys = False
        for k, v in filter_def.items():
            if k not in ignored_keys and v:
                has_other_keys = True
                break

        if civs and not types and not races and not has_other_keys:
            civ_match_mode = filter_def.get("civ_match_mode", "OR").upper()
            joiner = "または" if civ_match_mode == "OR" else "と"
            if len(civs) == 1:
                 joiner = "・"
            civ_names = joiner.join([CardTextResources.get_civilization_text(c) for c in civs])
            return civ_names + "の文明"

        # 再発防止: types が空のときに「クリーチャー」をデフォルトにしない。
        #   フィルターでタイプ未指定は「カード」(全タイプ対象)。
        #   CREATURE のみ指定時だけ「クリーチャー」、SPELL のみなら「呪文」、
        #   複数タイプ指定時は "/" 区切り、races 指定があればそれを優先する。
        # データ駆動アプローチ: 優先度に応じた単位辞書を利用
        from dm_toolkit.consts import CARD_TYPE_UNIT_MAP
        if not types:
            noun_str = "カード"  # タイプ未指定は全タイプ対象
        else:
            words = []
            for t in types:
                if t == consts.CardType.CREATURE.value: words.append("クリーチャー")
                elif t == consts.CardType.SPELL.value: words.append("呪文")
                elif t == consts.CardType.ELEMENT.value: words.append("エレメント")
                else: words.append(tr(t) if tr(t) else "カード")
            noun_str = "/".join(words)

        if races:
            noun_str = "/".join(races)

        return adj_str + noun_str
