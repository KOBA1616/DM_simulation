# -*- coding: utf-8 -*-
from typing import Any, List, Dict, Optional

# dm_ai_module may be an optional compiled module
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    m = None

def get_card_civilizations(card_data: Any) -> List[str]:
    """
    Returns a list of civilization names (e.g. ["FIRE", "NATURE"]) from card data.
    Handles C++ pybind11 objects, C++ CardDatabase objects, and legacy dicts.
    """
    if not card_data:
        return ["COLORLESS"]

    # Handle dict format (from JSON)
    if isinstance(card_data, dict):
        civs_data = card_data.get('civilizations', [])
        if isinstance(civs_data, list) and civs_data:
            return [str(c) if isinstance(c, str) else c for c in civs_data]
        return ["COLORLESS"]

    # Handle object format (from C++)
    if hasattr(card_data, 'civilizations') and card_data.civilizations:
        civs = []
        for c in card_data.civilizations:
            if hasattr(c, 'name'):
                civs.append(c.name)
            else:
                civs.append(str(c).split('.')[-1])
        return civs

    elif hasattr(card_data, 'civilization'):
        # Legacy singular
        c = card_data.civilization
        if hasattr(c, 'name'):
            return [c.name]
        return [str(c).split('.')[-1]]

    return ["COLORLESS"]

def get_card_civilization(card_data: Any) -> str:
    """
    Returns the primary civilization name as a string.
    If multiple, returns the first one.
    """
    civs = get_card_civilizations(card_data)
    if civs:
        return civs[0]
    return "COLORLESS"

def get_card_name_by_instance(game_state: Any, card_db: Dict[int, Any], instance_id: int) -> str:
    if not game_state or not m: return f"Inst_{instance_id}"

    try:
        # Assuming GameState has get_card_instance exposed
        inst = game_state.get_card_instance(instance_id)
        if inst:
            card_id = inst.card_id
            if card_id in card_db:
                card_def = card_db[card_id]
                # Support both dict and object formats
                return card_def['name'] if isinstance(card_def, dict) else card_def.name
    except Exception:
        pass

    return f"Inst_{instance_id}"

def get_card_name(card_def: Any) -> str:
    """Get card name from dict or object format."""
    if isinstance(card_def, dict):
        return card_def.get('name', 'Unknown')
    return getattr(card_def, 'name', 'Unknown')

def get_card_cost(card_def: Any) -> int:
    """Get card cost from dict or object format."""
    if isinstance(card_def, dict):
        return card_def.get('cost', 0)
    return getattr(card_def, 'cost', 0)

def get_card_power(card_def: Any) -> int:
    """Get card power from dict or object format."""
    if isinstance(card_def, dict):
        return card_def.get('power', 0)
    return getattr(card_def, 'power', 0)
