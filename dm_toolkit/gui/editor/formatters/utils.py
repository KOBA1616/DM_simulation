from typing import Dict, Any, Optional, List
from dm_toolkit.gui.editor.text_resources import CardTextResources

def format_civs(civs: List[str]) -> str:
    """Format a list of civilizations into a string (e.g. '火/自然' or '無色')."""
    if not civs:
        return "無色"
    return "/".join([CardTextResources.get_civilization_text(c) for c in civs])

def get_command_amount_with_fallback(command: Dict[str, Any], default: int = 1) -> Any:
    """
    Safely extract the amount/quantity from a command dictionary.
    Prioritizes 'look_count', then uses get_command_amount().
    If the final extracted value is less than or equal to 0, it falls back to the default value.
    """
    amount = command.get("look_count")
    if amount is not None:
        try:
            if int(amount) <= 0:
                return default
            return amount
        except ValueError:
            return amount

    amount = get_command_amount(command, default=0)
    try:
        if int(amount) <= 0:
            return default
    except ValueError:
        pass

    return amount

def get_command_amount(command: Dict[str, Any], default: int = 1) -> Any:
    """
    Safely extract the amount/quantity from a command dictionary.
    Prioritizes 'amount', then checks if it's within a 'filter'/'target_filter' count.
    """
    if "amount" in command and command["amount"] is not None:
        return command["amount"]

    filter_def = command.get("filter") or command.get("target_filter") or {}
    if isinstance(filter_def, dict) and "count" in filter_def and filter_def["count"] is not None:
        return filter_def["count"]

    from dm_toolkit.gui.editor.schema_def import get_schema
    schema = get_schema(command.get('type', ''))
    if schema and schema.default_amount is not None:
        return schema.default_amount

    return default

def get_command_max_cost(command: Dict[str, Any], default: Any = None) -> Any:
    """
    Safely extract the max_cost from a command dictionary or its target_filter.
    """
    max_cost_src = command.get('max_cost')
    if max_cost_src is None and 'target_filter' in command:
        max_cost_src = (command.get('target_filter') or {}).get('max_cost')

    if max_cost_src is not None and not isinstance(max_cost_src, dict):
        return max_cost_src

    return default
