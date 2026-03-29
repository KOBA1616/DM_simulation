from typing import Any, Optional
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

class ValueResolver:
    """
    Abstracts input link checking and provides natural text for values
    that could be numeric constants or input-linked pronouns (e.g., "その数").
    """

    @classmethod
    def resolve_range(cls, min_val: Any, max_val: Any, unit: str = "コスト", min_usage: str = "MIN_COST", max_usage: str = "MAX_COST", linked_token: str = "その数", has_input_key: bool = False, input_usage: str = "") -> str:
        """
        Formats a range description (e.g. "コスト3～5", "パワー5000以上").
        Returns the formatted string.
        """
        if min_val is None:
            min_val = 0
        default_max = 999 if unit == "コスト" else 999999
        if max_val is None:
            max_val = default_max

        is_min_linked = InputLinkFormatter.is_input_linked(min_val, usage=min_usage) or (has_input_key and input_usage == min_usage)
        is_max_linked = InputLinkFormatter.is_input_linked(max_val, usage=max_usage) or (has_input_key and input_usage == max_usage)

        if is_min_linked:
            return f"{unit}{linked_token}以上"
        if is_max_linked:
            return f"{unit}{linked_token}以下"

        min_n = min_val if isinstance(min_val, int) else 0
        max_n = max_val if isinstance(max_val, int) else default_max

        if min_n > 0 and max_n < default_max:
            return f"{unit}{min_n}～{max_n}"
        elif min_n > 0:
            return f"{unit}{min_n}以上"
        elif max_n < default_max:
            return f"{unit}{max_n}以下"

        return ""
