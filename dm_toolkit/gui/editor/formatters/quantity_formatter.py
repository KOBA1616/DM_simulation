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
    def apply_to_template(cls, template: str, formatted_qty: str, is_all: bool, up_to: bool, to_z: str, from_z: str, modifier: str = "") -> str:
        """
        Applies the formatted quantity to a template, adjusting verbs (選び、戻す、置く、出す)
        based on up_to and 'all' rules.
        Resolves the {modifier} token by safely placing it before verbs without string suffix checking.
        """
        # Common pattern replaces {amount}{unit} with the formatted quantity string
        result = template

        # Make sure the template has the {modifier} token directly where needed.
        # If the token is missing from the template dict, safely inject it before known verbs.
        # We prefer modifying text_resources.json natively, but provide a safe fallback just in case.
        if "{modifier}" not in result:
            for verb in ["置く", "出す", "加える", "戻す", "移動する"]:
                if f"{verb}。" in result:
                    result = result.replace(f"{verb}。", f"{{modifier}}{verb}。")
                elif f"、{verb}" in result:
                    result = result.replace(f"、{verb}", f"、{{modifier}}{verb}")

        if is_all:
            result = result.replace("{amount}{unit}", formatted_qty).replace("選び、", "")
            return result.replace("{modifier}", modifier)

        if up_to and formatted_qty != "すべて" and "最大" in formatted_qty:
            # Handle "up to" case ("まで選び")
            if to_z == "HAND" and from_z != "DECK":
                # We target only exact token combinations, avoiding generic string verb matching.
                # However, some hardcoded verbs still exist in `replace("戻す", "選び、戻す")`.
                # We refactor these to strictly target the final noun structure safely.
                result = result.replace("{amount}{unit}", formatted_qty)
                if "選び、" not in result:
                    result = result.replace("{to_z}", "選び、{to_z}")
            elif to_z in ["GRAVEYARD", "MANA_ZONE", "DECK_BOTTOM", "BATTLE_ZONE"]:
                result = result.replace("{amount}{unit}", formatted_qty)
                if "選び、" not in result:
                     result = result.replace("{to_z}", "選び、{to_z}")
            elif (from_z, to_z) == ("DECK", "HAND"):
                result = result.replace("{amount}{unit}", formatted_qty)

        # Base replacement for standard quantity
        result = result.replace("{amount}{unit}", formatted_qty)

        return result.replace("{modifier}", modifier)
