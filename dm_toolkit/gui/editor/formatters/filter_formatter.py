from typing import Dict, Any, List, Optional
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.formatters.utils import is_input_linked
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

        is_min_linked = is_input_linked(min_val, usage=min_usage)
        is_max_linked = is_input_linked(max_val, usage=max_usage)

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

        scope_text = CardTextResources.get_scope_text(scope)
        if not scope_text:
            return text

        scope_variants = []
        if scope in ("OPPONENT", "PLAYER_OPPONENT"):
            scope_variants.extend(["相手が", "相手の"])
        if scope in ("SELF", "PLAYER_SELF"):
            scope_variants.extend(["自分が", "自分の"])

        for v in scope_variants:
            if v in text:
                return text

        # Handle cases where scope_text is just "自分" or "相手" without "の"
        # and we need to attach it to a noun
        if not text:
            return scope_text

        if scope_text.endswith("の") and text.startswith("の"):
            return scope_text + text[1:]

        return f"{scope_text}{text}"
