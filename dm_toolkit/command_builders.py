# -*- coding: utf-8 -*-
"""
GameCommand Builder Utilities

This module provides helper functions to construct canonical GameCommand dictionaries
directly, supporting the gradual migration away from legacy Action dictionary patterns
as outlined in AGENTS.md Policy Section 1.

Philosophy:
- Prefer building commands directly over Action dict + map_action conversion
- Reduce overhead and complexity by using standardized builder functions
- Maintain clear, type-safe command construction patterns

Usage:
    from dm_toolkit.command_builders import build_draw_command, build_transition_command
    
    # Direct command construction
    draw_cmd = build_draw_command(amount=2)
    transition_cmd = build_transition_command(from_zone="HAND", to_zone="MANA", amount=1)
    
    # Execute via unified execution path
    from dm_toolkit.unified_execution import ensure_executable_command
    cmd = ensure_executable_command(draw_cmd)

Migration Path:
    Phase 1: Use alongside map_action for gradual transition
    Phase 2: Adopt in new test code and features
    Phase 3: Refactor existing tests to use builders
    Phase 4: Deprecate direct Action dict construction in tests
"""

from typing import Any, Dict, Optional, List
import uuid


def _ensure_uid(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure command has a unique identifier."""
    if 'uid' not in cmd:
        cmd['uid'] = str(uuid.uuid4())
    return cmd


def build_draw_command(
    from_zone: str = "DECK",
    to_zone: str = "HAND",
    amount: int = 1,
    owner_id: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized DRAW_CARD command.
    
    Args:
        from_zone: Source zone (default: DECK)
        to_zone: Destination zone (default: HAND)
        amount: Number of cards to draw (default: 1)
        owner_id: Optional player ID
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "DRAW_CARD",
        "from_zone": from_zone,
        "to_zone": to_zone,
        "amount": amount,
        **kwargs
    }
    if owner_id is not None:
        cmd["owner_id"] = owner_id
    return _ensure_uid(cmd)


def build_transition_command(
    from_zone: str,
    to_zone: str,
    amount: int = 1,
    owner_id: Optional[int] = None,
    source_instance_id: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized TRANSITION command for moving cards between zones.
    
    Args:
        from_zone: Source zone (e.g., HAND, DECK, BATTLE)
        to_zone: Destination zone (e.g., MANA, GRAVEYARD, SHIELD)
        amount: Number of cards to move (default: 1)
        owner_id: Optional player ID
        source_instance_id: Optional specific card instance ID
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "TRANSITION",
        "from_zone": from_zone,
        "to_zone": to_zone,
        "amount": amount,
        **kwargs
    }
    if owner_id is not None:
        cmd["owner_id"] = owner_id
    if source_instance_id is not None:
        cmd["source_instance_id"] = source_instance_id
    return _ensure_uid(cmd)


def build_mana_charge_command(
    source_instance_id: int,
    from_zone: str = "HAND",
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized MANA_CHARGE command.
    
    Args:
        source_instance_id: Card instance ID to charge as mana
        from_zone: Source zone (default: HAND)
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "MANA_CHARGE",
        "source_instance_id": source_instance_id,
        "from_zone": from_zone,
        "to_zone": "MANA",
        **kwargs
    }
    return _ensure_uid(cmd)


def build_destroy_command(
    source_instance_id: Optional[int] = None,
    from_zone: str = "BATTLE",
    target_filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized DESTROY command.
    
    Args:
        source_instance_id: Optional specific card instance to destroy
        from_zone: Source zone (default: BATTLE)
        target_filter: Optional filter for selecting targets
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "DESTROY",
        "from_zone": from_zone,
        "to_zone": "GRAVEYARD",
        **kwargs
    }
    if source_instance_id is not None:
        cmd["source_instance_id"] = source_instance_id
    if target_filter is not None:
        cmd["target_filter"] = target_filter
    return _ensure_uid(cmd)


def build_tap_command(
    source_instance_id: Optional[int] = None,
    target_filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized TAP command.
    
    Args:
        source_instance_id: Optional specific card instance to tap
        target_filter: Optional filter for selecting targets
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "TAP",
        **kwargs
    }
    if source_instance_id is not None:
        cmd["source_instance_id"] = source_instance_id
    if target_filter is not None:
        cmd["target_filter"] = target_filter
    return _ensure_uid(cmd)


def build_untap_command(
    source_instance_id: Optional[int] = None,
    target_filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized UNTAP command.
    
    Args:
        source_instance_id: Optional specific card instance to untap
        target_filter: Optional filter for selecting targets
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "UNTAP",
        **kwargs
    }
    if source_instance_id is not None:
        cmd["source_instance_id"] = source_instance_id
    if target_filter is not None:
        cmd["target_filter"] = target_filter
    return _ensure_uid(cmd)


def build_mutate_command(
    mutation_kind: str,
    amount: int = 0,
    source_instance_id: Optional[int] = None,
    target_filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized MUTATE command for modifying card properties.
    
    Args:
        mutation_kind: Type of mutation (e.g., POWER_MOD, COST, HEAL)
        amount: Mutation amount/value
        source_instance_id: Optional specific card instance to mutate
        target_filter: Optional filter for selecting targets
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "MUTATE",
        "mutation_kind": mutation_kind,
        "amount": amount,
        **kwargs
    }
    if source_instance_id is not None:
        cmd["source_instance_id"] = source_instance_id
    if target_filter is not None:
        cmd["target_filter"] = target_filter
    return _ensure_uid(cmd)


def build_attack_player_command(
    attacker_instance_id: int,
    target_player: int,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized ATTACK_PLAYER command.
    
    Args:
        attacker_instance_id: Attacking creature's instance ID
        target_player: Target player ID
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "ATTACK_PLAYER",
        "instance_id": attacker_instance_id,
        "target_player": target_player,
        **kwargs
    }
    return _ensure_uid(cmd)


def build_choice_command(
    options: List[List[Dict[str, Any]]],
    amount: int = 1,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Build a standardized CHOICE command for player selections.
    
    Args:
        options: List of option groups (each group is a list of commands)
        amount: Number of choices to make (default: 1)
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    cmd = {
        "type": "CHOICE",
        "options": options,
        "amount": amount,
        **kwargs
    }
    return _ensure_uid(cmd)
