from typing import Dict, Any, List
from dm_toolkit.consts import TargetScope

def normalize_card_data(card_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively normalizes legacy card data schema into the current canonical schema.
    This replaces scattershot normalization logic in UI and formatting layers.
    """
    if not isinstance(card_data, dict):
        return card_data
    
    _normalize_dict(card_data)

    # Walk through effects
    for effect in card_data.get("effects", []):
        if isinstance(effect, dict):
            _normalize_dict(effect)

            # Walk through commands
            for command in effect.get("commands", []):
                if isinstance(command, dict):
                    _normalize_dict(command)

    # Walk through static abilities
    for ability in card_data.get("static_abilities", []):
        if isinstance(ability, dict):
            _normalize_dict(ability)

    # Walk through cost reductions
    for cr in card_data.get("cost_reductions", []):
        if isinstance(cr, dict):
            _normalize_dict(cr)

    # Walk through metamorph abilities
    for meta in card_data.get("metamorph_abilities", []):
        if isinstance(meta, dict):
            _normalize_dict(meta)

            # Walk through commands
            for command in meta.get("commands", []):
                if isinstance(command, dict):
                    _normalize_dict(command)

    # Handle spell side recursively
    spell_side = card_data.get("spell_side")
    if isinstance(spell_side, dict):
        normalize_card_data(spell_side)

    return card_data


def _normalize_dict(data: Dict[str, Any]):
    """Applies field-level normalizations to a single dictionary."""
    if not isinstance(data, dict):
        return

    # 1. filter -> target_filter
    if "filter" in data and "target_filter" not in data:
        data["target_filter"] = data.pop("filter")

    # 2. trigger_filter -> target_filter
    if "trigger_filter" in data and "target_filter" not in data:
        data["target_filter"] = data.pop("trigger_filter")
        
    # 3. target_group -> scope
    if "target_group" in data and "scope" not in data:
        data["scope"] = data.pop("target_group")

    # Normalize scope to TargetScope
    if "scope" in data and isinstance(data["scope"], str):
        data["scope"] = TargetScope.normalize(data["scope"])

    # Nested target_filter needs its own normalization?
    tf = data.get("target_filter")
    if isinstance(tf, dict):
        _normalize_dict(tf)

    # Condition needs its own normalization
    cond = data.get("condition") or data.get("condition_def")
    if isinstance(cond, dict):
        _normalize_dict(cond)

    # REVOLUTION_CHANGE specific normalization
    if data.get("type") == "REVOLUTION_CHANGE" or data.get("name") == "REVOLUTION_CHANGE":
        if "target_filter" not in data and "filter" in data:
             data["target_filter"] = data.pop("filter")
