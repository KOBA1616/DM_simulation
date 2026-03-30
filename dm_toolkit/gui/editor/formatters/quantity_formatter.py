from typing import Any, Optional
from dm_toolkit.gui.editor.formatters.text_utils import TextUtils

class QuantityFormatter:
    """Formatter for handling quantities, 'all', and 'up to' semantics uniformly."""

    @classmethod
    def format_quantity(cls, amount: Any, unit: str, up_to: bool = False, is_all: bool = False, linked_text: Optional[str] = None) -> str:
        """
        Generates the proper Japanese phrase for a quantity, handling 'all', 'up to', and input links.

        Args:
            amount: The numeric or linked value.
            unit: The unit string (e.g., '枚', '体', 'つ').
            up_to: Whether the quantity represents an 'up to' limit.
            is_all: Whether the quantity means 'all' ('すべて').
            linked_text: If provided, this text (e.g., 'その数') replaces the amount+unit.
        """
        if is_all:
            return "すべて"

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
    def apply_to_template(cls, template: str, formatted_qty: str, is_all: bool, up_to: bool, to_z: str, from_z: str) -> str:
        """
        Applies the formatted quantity to a template, adjusting verbs (選び、戻す、置く、出す)
        based on up_to and 'all' rules.
        """
        # Common pattern replaces {amount}{unit} with the formatted quantity string
        result = template

        if is_all:
            result = result.replace("{amount}{unit}", formatted_qty).replace("選び、", "")
            return result

        if up_to and formatted_qty != "すべて" and "最大" in formatted_qty:
            # Handle "up to" case ("まで選び")
            if to_z == "HAND" and from_z != "DECK":
                result = result.replace("{amount}{unit}", formatted_qty).replace("戻す", "選び、戻す")
                if "選び、" not in result:
                    result = result.replace("まで{to_z}", "まで選び、{to_z}")
            elif to_z in ["GRAVEYARD", "MANA_ZONE", "DECK_BOTTOM", "BATTLE_ZONE"]:
                result = result.replace("{amount}{unit}", formatted_qty).replace("置く", "選び、置く").replace("出す", "選び、出す")
                if "選び、" not in result:
                     result = result.replace("まで{to_z}", "まで選び、{to_z}")
            elif (from_z, to_z) == ("DECK", "HAND"):
                result = result.replace("{amount}{unit}", formatted_qty)

        # Base replacement for standard quantity
        result = result.replace("{amount}{unit}", formatted_qty)

        return result
