from typing import Any, Optional
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils

class QuantityFormatter:
    """Formatter for handling quantities, 'all', and 'up to' semantics uniformly."""

    @classmethod
    def format_quantity(cls, amount: Any, unit: str, up_to: bool = False, is_all: bool = False, linked_text: Optional[str] = None, targeting_mode: str = "TARGET") -> str:
        """
        Generates the proper Japanese phrase for a quantity, handling 'all', 'up to', and input links.

        Args:
            amount: The numeric or linked value.
            unit: The unit string (e.g., '枚', '体', 'つ').
            up_to: Whether the quantity represents an 'up to' limit.
            is_all: Whether the quantity means 'all' ('すべて').
            linked_text: If provided, this text (e.g., 'その数') replaces the amount+unit.
            targeting_mode: The targeting mode string.
        """
        # For the new QuantityModeWidget, amount == -1 signifies "All"
        if is_all or amount == -1:
            return "すべて"

        if targeting_mode == "NON_TARGET":
            return "すべての"

        if linked_text:
            if up_to:
                if linked_text.endswith("だけ"):
                    return f"最大{linked_text}まで"
                return f"最大{linked_text}まで"
            return linked_text

        # Using TextUtils format_up_to for consistent "最大〜まで" structure
        if up_to:
            return TextUtils.format_up_to(amount, unit, up_to=True)

        return f"{amount}{unit}"

    @classmethod
    def format_selection_quantity(cls, count: Any, unit: str) -> str:
        """Format the number of cards implicitly selected by a filter."""
        if isinstance(count, int) and count > 1:
            return f"{count}{unit}まで"
        return f"1{unit}"

    @classmethod
    def apply_to_template(cls, template: str, formatted_qty: str, is_all: bool, up_to: bool, to_z: str, from_z: str, modifier: str = "", targeting_mode: str = "NON_TARGET") -> str:
        """
        Applies the formatted quantity to a template, adjusting verbs (選び、戻す、置く、出す)
        based on up_to and 'all' rules.
        Resolves the {modifier} token by safely placing it before verbs without string suffix checking.
        """
        # Common pattern replaces {amount}{unit} with the formatted quantity string
        result = template

        # When TargetingMode is NON_TARGET, it implicitly behaves like "all matching without selecting"
        if is_all or targeting_mode == "NON_TARGET":
            if targeting_mode == "NON_TARGET" and not is_all and "{amount}{unit}" in result:
                # If NON_TARGET but a specific quantity is requested, it means "those N cards"
                pass
            else:
                if targeting_mode == "NON_TARGET" and not is_all:
                    formatted_qty = "すべての"
                # Replace selection verb with nothing if it's NON_TARGET or ALL
                result = result.replace("選び、", "")

        if is_all:
            result = result.replace("{amount}{unit}", formatted_qty).replace("選び、", "")
            return result.replace("{modifier}", modifier)

        # Delegate selection verb injection to the template via {selection_verb}
        # to avoid hardcoded string replacement logic on zone variables.
        selection_verb = "選び、" if targeting_mode == "TARGET" else ""
        result = result.replace("{selection_verb}", selection_verb)

        # Legacy fallback if {selection_verb} isn't used yet, ensure the verb gets prepended to modifier instead
        if targeting_mode == "TARGET" and "選び、" not in result:
            if not modifier.startswith("選び、"):
                modifier = selection_verb + modifier

        # Base replacement for standard quantity
        # If targeting_mode is NON_TARGET and we modified formatted_qty to "すべての",
        # it might need to prefix the target instead. This requires text generator cooperation,
        # but for now we replace the amount/unit placeholder.
        result = result.replace("{amount}{unit}", formatted_qty)

        return result.replace("{modifier}", modifier)
