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

from typing import Any, Dict, Optional, List, Union
import uuid

# Try to import dm_ai_module to get enums and classes
try:
    import dm_ai_module
    _CommandType = dm_ai_module.CommandType
    _CommandDef = dm_ai_module.CommandDef
    _FilterDef = dm_ai_module.FilterDef
    _ConditionDef = getattr(dm_ai_module, "ConditionDef", None)
    _HAS_NATIVE = True
except ImportError:
    dm_ai_module = None
    _CommandType = None
    _CommandDef = None
    _FilterDef = None
    _ConditionDef = None
    _HAS_NATIVE = False


def _ensure_uid(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure command has a unique identifier."""
    if 'uid' not in cmd:
        cmd['uid'] = str(uuid.uuid4())
    return cmd


def _build_filter_def(tf_data: Dict[str, Any]) -> Any:
    """Helper to build a FilterDef from a dict."""
    if not _HAS_NATIVE:
        return None
    f = _FilterDef()
    for k in ['zones', 'types', 'civilizations', 'races', 'min_cost', 'max_cost',
              'exact_cost', 'min_power', 'max_power', 'is_tapped', 'is_blocker',
              'is_evolution', 'owner', 'count', 'cost_ref', 'selection_mode']:
        if k in tf_data:
            val = tf_data[k]
            # Special handling for civilizations enum conversion
            if k == 'civilizations' and isinstance(val, list):
                converted_civs = []
                for c in val:
                    if isinstance(c, str) and hasattr(dm_ai_module.Civilization, c):
                        converted_civs.append(getattr(dm_ai_module.Civilization, c))
                    else:
                        converted_civs.append(c)

                # Pybind11 bind_vector requires explicit type for CivilizationList in some cases
                if hasattr(dm_ai_module, 'CivilizationList'):
                    c_list = dm_ai_module.CivilizationList()
                    for c in converted_civs:
                        c_list.append(c)
                    setattr(f, k, c_list)
                else:
                    setattr(f, k, converted_civs)
            else:
                setattr(f, k, val)

    if 'and_conditions' in tf_data:
        for sub in tf_data['and_conditions']:
            if isinstance(sub, dict):
                f.and_conditions.append(_build_filter_def(sub))

    return f


def _build_condition_def(cond_data: Dict[str, Any]) -> Any:
    """Helper to build a ConditionDef from a dict."""
    if not _HAS_NATIVE or not _ConditionDef:
        return None
    c = _ConditionDef()
    if 'type' in cond_data: c.type = cond_data['type']
    if 'value' in cond_data: c.value = cond_data['value']
    if 'str_val' in cond_data: c.str_val = cond_data['str_val']
    if 'stat_key' in cond_data: c.stat_key = cond_data['stat_key']
    if 'op' in cond_data: c.op = cond_data['op']
    if 'filter' in cond_data and cond_data['filter']:
        if isinstance(cond_data['filter'], dict):
            c.filter = _build_filter_def(cond_data['filter'])
        elif isinstance(cond_data['filter'], _FilterDef):
            c.filter = cond_data['filter']
    return c


def _build_native_command(cmd_type_str: str, **kwargs: Any) -> Any:
    """Helper to build a native CommandDef object."""
    if not _HAS_NATIVE:
        raise ImportError("dm_ai_module is not available, cannot build native command.")

    cmd = _CommandDef()

    # Set Type
    if hasattr(_CommandType, cmd_type_str):
        cmd.type = getattr(_CommandType, cmd_type_str)

    # Map common kwargs to CommandDef fields
    if 'instance_id' in kwargs and kwargs['instance_id'] is not None: cmd.instance_id = kwargs['instance_id']
    if 'source_instance_id' in kwargs and kwargs['source_instance_id'] is not None: cmd.instance_id = kwargs['source_instance_id'] # alias
    if 'target_instance' in kwargs and kwargs['target_instance'] is not None: cmd.target_instance = kwargs['target_instance']
    if 'owner_id' in kwargs and kwargs['owner_id'] is not None: cmd.owner_id = kwargs['owner_id']
    if 'amount' in kwargs: cmd.amount = kwargs['amount']
    if 'from_zone' in kwargs: cmd.from_zone = kwargs['from_zone']
    if 'to_zone' in kwargs: cmd.to_zone = kwargs['to_zone']
    if 'mutation_kind' in kwargs: cmd.mutation_kind = kwargs['mutation_kind']
    if 'str_param' in kwargs: cmd.str_param = kwargs['str_param']
    if 'optional' in kwargs: cmd.optional = kwargs['optional']
    if 'up_to' in kwargs: cmd.up_to = kwargs['up_to']

    if 'input_value_key' in kwargs: cmd.input_value_key = kwargs['input_value_key']
    if 'input_value_usage' in kwargs: cmd.input_value_usage = kwargs['input_value_usage']
    if 'output_value_key' in kwargs: cmd.output_value_key = kwargs['output_value_key']
    if 'slot_index' in kwargs: cmd.slot_index = kwargs['slot_index']
    if 'target_slot_index' in kwargs: cmd.target_slot_index = kwargs['target_slot_index']

    # Handle Target Group
    if 'target_group' in kwargs and kwargs['target_group'] is not None:
        tg = kwargs['target_group']
        if isinstance(tg, int): # Enum value
             cmd.target_group = tg
        elif isinstance(tg, str) and hasattr(dm_ai_module.TargetScope, tg):
            cmd.target_group = getattr(dm_ai_module.TargetScope, tg)

    # Handle Target Filter
    if 'target_filter' in kwargs and kwargs['target_filter']:
        tf_data = kwargs['target_filter']
        if isinstance(tf_data, dict):
            cmd.target_filter = _build_filter_def(tf_data)
        elif isinstance(tf_data, _FilterDef):
             cmd.target_filter = tf_data

    # Handle Condition
    if 'condition' in kwargs and kwargs['condition']:
        c_data = kwargs['condition']
        if isinstance(c_data, dict):
            cmd.condition = _build_condition_def(c_data)
        elif _ConditionDef and isinstance(c_data, _ConditionDef):
            cmd.condition = c_data

    # Handle Recursive Command Lists (if_true, if_false)
    if 'if_true' in kwargs and kwargs['if_true']:
        true_cmds = []
        for item in kwargs['if_true']:
            if isinstance(item, _CommandDef):
                true_cmds.append(item)
            elif isinstance(item, dict):
                ctype = item.get('type', 'NONE')
                item_kwargs = item.copy()
                if 'type' in item_kwargs: del item_kwargs['type']
                true_cmds.append(_build_native_command(ctype, **item_kwargs))
        cmd.if_true = true_cmds

    if 'if_false' in kwargs and kwargs['if_false']:
        false_cmds = []
        for item in kwargs['if_false']:
            if isinstance(item, _CommandDef):
                false_cmds.append(item)
            elif isinstance(item, dict):
                ctype = item.get('type', 'NONE')
                item_kwargs = item.copy()
                if 'type' in item_kwargs: del item_kwargs['type']
                false_cmds.append(_build_native_command(ctype, **item_kwargs))
        cmd.if_false = false_cmds

    # Handle Options (List[List[CommandDef]])
    if 'options' in kwargs and kwargs['options']:
        options_list = []
        for opt_group in kwargs['options']:
            native_group = []
            for item in opt_group:
                if isinstance(item, _CommandDef):
                    native_group.append(item)
                elif isinstance(item, dict):
                    ctype = item.get('type', 'NONE')
                    item_kwargs = item.copy()
                    if 'type' in item_kwargs: del item_kwargs['type']
                    native_group.append(_build_native_command(ctype, **item_kwargs))
            options_list.append(native_group)
        # Note: CommandDef must support 'options' field (std::vector<std::vector<CommandDef>>)
        if hasattr(cmd, 'options'):
            cmd.options = options_list

    return cmd


def build_draw_command(
    from_zone: str = "DECK",
    to_zone: str = "HAND",
    amount: int = 1,
    owner_id: Optional[int] = None,
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized DRAW_CARD command.
    
    Args:
        from_zone: Source zone (default: DECK)
        to_zone: Destination zone (default: HAND)
        amount: Number of cards to draw (default: 1)
        owner_id: Optional player ID
        native: If True, returns a native CommandDef object (requires dm_ai_module)
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("DRAW_CARD", from_zone=from_zone, to_zone=to_zone,
                                     amount=amount, owner_id=owner_id, **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized TRANSITION command for moving cards between zones.
    
    Args:
        from_zone: Source zone (e.g., HAND, DECK, BATTLE)
        to_zone: Destination zone (e.g., MANA, GRAVEYARD, SHIELD)
        amount: Number of cards to move (default: 1)
        owner_id: Optional player ID
        source_instance_id: Optional specific card instance ID
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("TRANSITION", from_zone=from_zone, to_zone=to_zone,
                                     amount=amount, owner_id=owner_id,
                                     source_instance_id=source_instance_id, **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized MANA_CHARGE command.
    
    Args:
        source_instance_id: Card instance ID to charge as mana
        from_zone: Source zone (default: HAND)
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("MANA_CHARGE", source_instance_id=source_instance_id,
                                     from_zone=from_zone, to_zone="MANA", **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized DESTROY command.
    
    Args:
        source_instance_id: Optional specific card instance to destroy
        from_zone: Source zone (default: BATTLE)
        target_filter: Optional filter for selecting targets
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("DESTROY", source_instance_id=source_instance_id,
                                     from_zone=from_zone, to_zone="GRAVEYARD",
                                     target_filter=target_filter, **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized TAP command.
    
    Args:
        source_instance_id: Optional specific card instance to tap
        target_filter: Optional filter for selecting targets
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("TAP", source_instance_id=source_instance_id,
                                     target_filter=target_filter, **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized UNTAP command.
    
    Args:
        source_instance_id: Optional specific card instance to untap
        target_filter: Optional filter for selecting targets
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("UNTAP", source_instance_id=source_instance_id,
                                     target_filter=target_filter, **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized MUTATE command for modifying card properties.
    
    Args:
        mutation_kind: Type of mutation (e.g., POWER_MOD, COST, HEAL)
        amount: Mutation amount/value
        source_instance_id: Optional specific card instance to mutate
        target_filter: Optional filter for selecting targets
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("MUTATE", mutation_kind=mutation_kind, amount=amount,
                                     source_instance_id=source_instance_id,
                                     target_filter=target_filter, **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized ATTACK_PLAYER command.
    
    Args:
        attacker_instance_id: Attacking creature's instance ID
        target_player: Target player ID
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("ATTACK_PLAYER", instance_id=attacker_instance_id,
                                     owner_id=target_player, **kwargs)

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
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized CHOICE command for player selections.
    
    Args:
        options: List of option groups (each group is a list of commands)
        amount: Number of choices to make (default: 1)
        **kwargs: Additional command fields
        
    Returns:
        GameCommand dictionary ready for execution
    """
    if native:
        return _build_native_command("CHOICE", options=options, amount=amount, **kwargs)

    cmd = {
        "type": "CHOICE",
        "options": options,
        "amount": amount,
        **kwargs
    }
    return _ensure_uid(cmd)

def build_play_card_command(
    card_id: int,
    source_instance_id: int,
    from_zone: str = "HAND",
    to_zone: str = "BATTLE",
    amount: int = 1,
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized PLAY_FROM_ZONE command.

    Args:
        card_id: Card ID (for reference, usually tracked by instance)
        source_instance_id: Card instance ID to play
        from_zone: Source zone (default: HAND)
        to_zone: Destination zone (default: BATTLE)
        amount: Number of cards to play (usually 1)
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields

    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("PLAY_FROM_ZONE", instance_id=source_instance_id,
                                     from_zone=from_zone, to_zone=to_zone,
                                     amount=amount, **kwargs)

    cmd = {
        "type": "PLAY_FROM_ZONE",
        "instance_id": source_instance_id,
        "from_zone": from_zone,
        "to_zone": to_zone,
        "amount": amount,
        "card_id": card_id, # Optional legacy info
        **kwargs
    }
    return _ensure_uid(cmd)


def build_pass_command(
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized PASS command.

    Args:
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields

    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("PASS", **kwargs)

    cmd = {
        "type": "PASS",
        **kwargs
    }
    return _ensure_uid(cmd)


def build_shield_burn_command(
    amount: int,
    owner_id: Optional[int] = None,
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized SHIELD_BURN command.

    Args:
        amount: Number of shields to burn
        owner_id: Optional player ID
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields

    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("SHIELD_BURN", amount=amount, owner_id=owner_id, **kwargs)

    cmd = {
        "type": "SHIELD_BURN",
        "amount": amount,
        **kwargs
    }
    if owner_id is not None:
        cmd["owner_id"] = owner_id
    return _ensure_uid(cmd)


def build_discard_command(
    amount: int,
    from_zone: str = "HAND",
    owner_id: Optional[int] = None,
    native: bool = False,
    **kwargs: Any
) -> Union[Dict[str, Any], Any]:
    """
    Build a standardized DISCARD command.

    Args:
        amount: Number of cards to discard
        from_zone: Source zone (default: HAND)
        owner_id: Optional player ID
        native: If True, returns a native CommandDef object
        **kwargs: Additional command fields

    Returns:
        GameCommand dictionary or CommandDef object
    """
    if native:
        return _build_native_command("DISCARD", amount=amount, from_zone=from_zone,
                                     to_zone="GRAVEYARD", owner_id=owner_id, **kwargs)

    cmd = {
        "type": "DISCARD",
        "amount": amount,
        "from_zone": from_zone,
        "to_zone": "GRAVEYARD",
        **kwargs
    }
    if owner_id is not None:
        cmd["owner_id"] = owner_id
    return _ensure_uid(cmd)
