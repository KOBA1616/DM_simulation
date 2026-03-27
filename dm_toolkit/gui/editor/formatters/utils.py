from typing import Dict, Any, Optional

def is_input_linked(val: Any, usage: Optional[str] = None) -> bool:
    """
    Check if a value represents an input-linked variable.
    Usually represented as a dict with 'input_value_usage' or 'input_link'.
    If 'usage' is provided, it checks if it matches that specific usage.
    """
    if not isinstance(val, dict):
        return False

    if usage is not None:
        return val.get("input_value_usage") == usage

    return "input_value_usage" in val or "input_link" in val

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

    return default
