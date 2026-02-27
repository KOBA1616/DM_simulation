# -*- coding: utf-8 -*-
"""
Unified Action-to-Command Converter
    ...
    -> None:
This module serves as the **single source of truth** for converting legacy Action
dictionaries to standardized GameCommand structures.
    ...
    -> None:
Key Principles (Specs/AGENTS.md Policy):
1. All Action-to-Command conversions MUST go through this module's `map_action` function.
    ...
    -> None:
3. Maintains backward compatibility via `compat_wrappers.add_aliases_to_command`.
4. Eliminates ad-hoc dictionary manipulation in test code and wrappers.

Usage:
    from dm_toolkit.action_to_command import map_action
    
    legacy_action = {"type": "DRAW_CARD", "value1": 2}
    command = map_action(legacy_action)
    # command = {"type": "DRAW_CARD", "from_zone": "DECK", "to_zone": "HAND", "amount": 2, ...}
"""
import uuid
import copy
import os
import warnings
from typing import Any, Dict, List, Optional, cast
from dm_toolkit.compat_wrappers import add_aliases_to_command
from dm_toolkit.consts import COMMAND_TYPES

# Try to import dm_ai_module to get enums, otherwise define mocks/None.
# Note: In some environments the compiled extension may be missing or fail to load.
try:
    import dm_ai_module  # type: ignore
except Exception:
    dm_ai_module = None  # type: ignore

_CommandType = getattr(dm_ai_module, 'CommandType', None) if dm_ai_module is not None else None
_Zone = getattr(dm_ai_module, 'Zone', None) if dm_ai_module is not None else None

# Value used to represent "ALL" when amount is 0 for targeting commands
AMOUNT_ALL = 255

# Virtual command types that are allowed even if the native CommandType enum
# does not expose them (handled by Python-side fallbacks or post-processing).
_ALLOWED_VIRTUAL_COMMAND_TYPES = {"REPLACE_CARD_MOVE", "CHOICE"}

# Optional deprecation flag: enable via environment variable to surface guidance
_DEPRECATE_ACTION_DICTS = bool(os.getenv('DM_TOOLKIT_DEPRECATE_ACTION_DICTS'))

def enable_action_deprecation(flag: bool) -> None:
    """Enable or disable deprecation warnings for legacy Action dict conversion."""
    global _DEPRECATE_ACTION_DICTS
    _DEPRECATE_ACTION_DICTS = bool(flag)

def set_command_type_enum(enum_cls: Any) -> None:
    """
    Helper to inject a CommandType enum for testing or manual setup.
    """
    global _CommandType
    _CommandType = enum_cls

def normalize_action_zone_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1: Normalize legacy zone key variations to canonical keys.
    
    Ensures incoming action dictionaries have consistent zone keys by creating
    aliases for common variations:
    - from_zone -> source_zone (for compatibility)
    - to_zone -> destination_zone (for compatibility)
    
    This normalization allows downstream code to reliably access zone information
    without checking multiple key names.
    
    Args:
        data: Legacy action dictionary potentially using varied zone key names.
        
    Returns:
        Normalized dictionary with canonical zone keys added (original keys preserved).
    """
    if not isinstance(data, dict): 
        return data
    new_data = data.copy()
    # Add canonical keys if only legacy variants exist
    if 'source_zone' not in new_data and 'from_zone' in new_data:
        new_data['source_zone'] = new_data['from_zone']
    if 'destination_zone' not in new_data and 'to_zone' in new_data:
        new_data['destination_zone'] = new_data['to_zone']
    return new_data

def _get_zone(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d: return cast(Optional[str], d[k])
    return None

def _get_any(d: Dict[str, Any], keys: List[str]) -> Any:
    """Helper to get the first non-None value from a list of keys."""
    for k in keys:
        val = d.get(k)
        if val is not None:
            return val
    return None

def _transfer_targeting(act: Dict[str, Any], cmd: Dict[str, Any]) -> None:
    scope = act.get('scope', 'NONE')
    if scope == 'NONE' and 'filter' in act:
         scope = 'TARGET_SELECT'
    cmd['target_group'] = scope
    if 'filter' in act:
        cmd['target_filter'] = copy.deepcopy(act['filter'])
    if act.get('optional', False):
        flags = cast(List[Any], cmd.setdefault('flags', []))
        flags.append('OPTIONAL')

def _transfer_common_move_fields(act: Dict[str, Any], cmd: Dict[str, Any]) -> None:
    _transfer_targeting(act, cmd)
    if 'amount' in act:
         cmd['amount'] = act['amount']
    elif 'filter' in act and isinstance(act['filter'], dict) and 'count' in act['filter']:
         cmd['amount'] = act['filter']['count']
    elif 'value1' in act:
         cmd['amount'] = act['value1']

def _finalize_command(cmd: Dict[str, Any], act: Dict[str, Any]) -> None:
    """
    Phase 3: Finalize command with canonical field normalization.
    
    Applies final standardizations to ensure the command has all required fields
    in their canonical forms. Legacy keys are preserved for backward compatibility.
    
    Canonical Field Mappings:
    - str_val -> str_param (preferred by engine)
    - value1 -> amount (for count/quantity operations)
    - value2 -> preserved as-is for secondary parameters
    - flags -> propagated from action if present
    
    Args:
        cmd: Command dictionary to finalize (modified in-place).
        act: Original action dictionary (source for missing fields).
    """
    # Ensure every command has a unique identifier
    if 'uid' not in cmd:
        cmd['uid'] = str(uuid.uuid4())

    # Ensure amount/flags exist
    if 'amount' in act and 'amount' not in cmd:
        try:
            val = act.get('amount')
            if val is None:
                val = act.get('value1')
            cmd['amount'] = int(val if val is not None else 0)
        except Exception:
             pass # Ignore if not convertible

    # ------------------------------------------------------------------
    # Canonical Key Normalization (Specs/AGENTS.md Policy Section 2)
    # Copy legacy keys to canonical forms; preserve original for compatibility
    # ------------------------------------------------------------------
    
    # String parameter normalization: str_val -> str_param
    if 'str_param' not in cmd:
        if 'str_val' in cmd and cmd.get('str_val') is not None:
            cmd['str_param'] = cmd.get('str_val')
        elif 'str_param' in act and act.get('str_param') is not None:
            cmd['str_param'] = act.get('str_param')
        elif 'str_val' in act and act.get('str_val') is not None:
            cmd['str_param'] = act.get('str_val')

    # Primary numeric parameter normalization: value1 -> amount
    if 'amount' not in cmd:
        # Typical legacy payloads store primary numeric in value1
        if 'value1' in cmd and cmd.get('value1') is not None:
            cmd['amount'] = cmd.get('value1')
        elif 'amount' in act and act.get('amount') is not None:
            cmd['amount'] = act.get('amount')
        elif 'value1' in act and act.get('value1') is not None:
            cmd['amount'] = act.get('value1')

    # Secondary numeric parameter: preserve value2 explicitly
    if 'value2' not in cmd and 'value2' in act:
        cmd['value2'] = act.get('value2')

    # Choice/select flow flags: propagate from action
    if 'flags' not in cmd and isinstance(act.get('flags'), list):
        cmd['flags'] = list(act.get('flags') or [])

def _validate_command_type(cmd: Dict[str, Any]) -> None:
    """
    Phase 2: Strict Type Enforcement.
    If dm_ai_module is available, ensures cmd['type'] exists in CommandType.
    """
    if _CommandType is None:
        return

    ctype = cmd.get('type', 'NONE')
    if ctype in _ALLOWED_VIRTUAL_COMMAND_TYPES:
        return
    # If the command originated as the same legacy type, allow it through
    if cmd.get('legacy_original_type') == ctype:
        return

    # Two modes:
    # - Native enum present (dm_ai_module): be permissive; some builds expose a subset.
    # - Test-injected enum: be strict; tests use this to validate warning behavior.
    is_native_enum = bool(_CommandType is not None and getattr(_CommandType, '__module__', '') == 'dm_ai_module')

    if ctype in _ALLOWED_VIRTUAL_COMMAND_TYPES:
        return

    if is_native_enum:
        if ctype in COMMAND_TYPES:
            return
        if hasattr(_CommandType, ctype):
            return
    else:
        if _CommandType is not None and hasattr(_CommandType, ctype):
            return

    cmd['legacy_warning'] = True
    cmd.setdefault('legacy_invalid_type', ctype)

def map_action(action_data: Any) -> Dict[str, Any]:
    """
    Pure function to convert a legacy Action dictionary/object to a Command dictionary.
    """
    # Optional deprecation notice (configurable via env var or helper)
    if _DEPRECATE_ACTION_DICTS:
        try:
            warnings.warn(
                "map_action() is deprecated for new code: prefer dm_toolkit.command_builders + "
                "dm_toolkit.unified_execution.ensure_executable_command.",
                DeprecationWarning,
            )
        except Exception:
            pass

    # Phase 1: Native Command Pass-through
    if dm_ai_module and hasattr(dm_ai_module, 'CommandDef') and isinstance(action_data, dm_ai_module.CommandDef):
        return action_data.to_dict()

    # Safe Copy
    try:
        if hasattr(action_data, 'to_dict'):
            act_data = action_data.to_dict()
        elif hasattr(action_data, '__dict__'):
            act_data = action_data.__dict__.copy()
        elif isinstance(action_data, dict):
            act_data = copy.deepcopy(action_data)
        else:
             return _create_error_command(str(action_data), "Invalid action shape")
    except Exception:
         return _create_error_command(str(action_data), "Uncopyable action")

    act_data = normalize_action_zone_keys(act_data)

    # Extract Type
    raw_type = act_data.get('type', 'NONE')
    act_type = str(raw_type).upper()
    if hasattr(raw_type, 'name'):
        act_type = raw_type.name.upper()    

    cmd = {
        "type": "NONE",
        "uid": str(uuid.uuid4()),
        "legacy_warning": False,
        "legacy_original_type": act_type  # Phase C: Track original type for metrics
    }

    # Common Fields
    for key in ['input_value_key', 'output_value_key', 'uid']:
        if key in act_data:
            cmd[key] = act_data[key]

    # Preserve slot_index when provided (e.g., RESOLVE_EFFECT selection)
    if 'slot_index' in act_data:
        cmd['slot_index'] = act_data['slot_index']

    # Preserve source_instance_id for generic filtering before specific handlers override
    if 'source_instance_id' in act_data and 'instance_id' not in cmd:
        cmd['instance_id'] = act_data['source_instance_id']

    # Preserve existing instance_id (idempotency for double-mapping)
    if 'instance_id' in act_data and 'instance_id' not in cmd:
        cmd['instance_id'] = act_data['instance_id']

    # Recursion (Options)
    if 'options' in act_data and isinstance(act_data['options'], list):
        cmd['options'] = []
        for opt in act_data['options']:
            options_list = cast(List[Any], cmd['options'])
            if isinstance(opt, list):
                options_list.append([map_action(sub) for sub in opt])
            else:
                options_list.append([map_action(opt)])

    src = _get_zone(act_data, ['source_zone', 'from_zone', 'origin_zone'])
    dest = _get_zone(act_data, ['destination_zone', 'to_zone', 'dest_zone'])

    # Phase 4.2: Normalize Zone Names to Enum
    # Mappings from Legacy/Common strings to dm_ai_module.Zone Enum names
    # Normalize various legacy zone names to short canonical forms used by tests/engine
    zone_map = {
        "MANA_ZONE": "MANA",
        "MANA": "MANA",
        "BATTLE_ZONE": "BATTLE",
        "BATTLE": "BATTLE",
        "SHIELD_ZONE": "SHIELD",
        "SHIELD": "SHIELD",
        "GRAVEYARD": "GRAVEYARD",
        "HAND": "HAND",
        "DECK": "DECK",
        "DECK_BOTTOM": "DECK_BOTTOM",
        "BUFFER": "BUFFER",
        "UNDER_CARD": "UNDER_CARD",
    }

    if src and src in zone_map:
        src = zone_map[src]
    if dest and dest in zone_map:
        dest = zone_map[dest]

    # --- Logic Mapping ---

    if act_type == "REPLACE_CARD_MOVE":
        _handle_replace_card_move(act_data, cmd, src, dest)

    elif act_type == "MOVE_CARD":
        _handle_move_card(act_data, cmd, src, dest)

    elif act_type in ["DESTROY", "DISCARD", "MANA_CHARGE", "RETURN_TO_HAND",
                      "SEND_TO_MANA", "SEND_TO_DECK_BOTTOM", "ADD_SHIELD", "SHIELD_BURN",
                      "ADD_MANA", "SEARCH_DECK_BOTTOM", "MOVE_TO_UNDER_CARD"]:
        _handle_specific_moves(act_type, act_data, cmd, src)

    elif act_type == "DRAW_CARD":
        # Preserve DRAW_CARD type for compatibility
        cmd['type'] = act_type
        cmd['from_zone'] = src or 'DECK'
        cmd['to_zone'] = dest or 'HAND'
        _transfer_common_move_fields(act_data, cmd)

    elif act_type in ["TAP", "UNTAP"]:
        cmd['type'] = act_type
        _transfer_targeting(act_data, cmd)
        if 'amount' in act_data: cmd['amount'] = act_data['amount']
        elif 'value1' in act_data: cmd['amount'] = act_data['value1']
        if 'amount' in cmd and cmd['amount'] == 0:
            cmd['amount'] = AMOUNT_ALL

    elif act_type in ["COUNT_CARDS", "MEASURE_COUNT", "GET_GAME_STAT"]:
        cmd['type'] = "QUERY"
        # Provide a structured query_kind for downstream handling
        if 'COUNT' in act_type:
            cmd['query_kind'] = 'COUNT_MATCHING'
            cmd['str_param'] = act_data.get('str_val', 'CARDS_MATCHING_FILTER')
        else:
            cmd['query_kind'] = 'GET_GAME_STAT'
            cmd['str_param'] = act_data.get('str_val', '')
        _transfer_targeting(act_data, cmd)

    elif act_type in ["APPLY_MODIFIER", "COST_REDUCTION", "GRANT_KEYWORD"]:
        _handle_modifiers(act_type, act_data, cmd)

    elif act_type in ["MUTATE", "POWER_MOD", "MODIFY_POWER"]:
        _handle_mutate(act_type, act_data, cmd)

    elif act_type in ["SELECT_OPTION", "SELECT_NUMBER", "SELECT_TARGET"]:
        _handle_selection(act_type, act_data, cmd)

    elif act_type in ["SEARCH_DECK", "SHUFFLE_DECK", "REVEAL_CARDS", "LOOK_AND_ADD", "MEKRAID", "REVOLUTION_CHANGE"]:
        # Keep legacy command type for compatibility; handler may add unified fields
        _handle_complex(act_type, act_data, cmd, dest)

    elif act_type in ["PLAY_CARD", "PLAY_FROM_ZONE", "FRIEND_BURST", "REGISTER_DELAYED_EFFECT", "CAST_SPELL", "DECLARE_PLAY"]:
        _handle_play_flow(act_type, act_data, cmd, src, dest)

    elif act_type == "RESET_INSTANCE":
        cmd['type'] = "MUTATE"
        cmd['mutation_kind'] = "RESET_INSTANCE"
        _transfer_targeting(act_data, cmd)

    elif act_type == "SEND_SHIELD_TO_GRAVE":
        # Explicit mapping for shield -> grave transitions
        cmd['type'] = 'TRANSITION'
        # Use legacy zone naming
        cmd['from_zone'] = act_data.get('source_zone') or act_data.get('from_zone') or 'SHIELD_ZONE'
        cmd['to_zone'] = act_data.get('destination_zone') or act_data.get('to_zone') or 'GRAVEYARD'
        if 'value1' in act_data: cmd['amount'] = act_data['value1']
        _transfer_common_move_fields(act_data, cmd)

    elif act_type == 'PUT_CREATURE':
        # Put creature directly into battle (not a normal play)
        cmd['type'] = 'TRANSITION'
        # Use legacy zone naming
        cmd['to_zone'] = act_data.get('destination_zone') or act_data.get('to_zone') or 'BATTLE_ZONE'
        if src and src != "NONE":
             cmd['from_zone'] = src
        if 'value1' in act_data: cmd['amount'] = act_data['value1']
        if 'str_val' in act_data: cmd['str_param'] = act_data['str_val']
        _transfer_common_move_fields(act_data, cmd)

    elif act_type == 'COST_REFERENCE':
        # Reference cost modifiers / markers
        cmd['type'] = 'MUTATE'
        cmd['mutation_kind'] = 'COST_REFERENCE'
        if 'value1' in act_data: cmd['amount'] = act_data['value1']
        if 'str_val' in act_data: cmd['str_param'] = act_data['str_val']
        _transfer_targeting(act_data, cmd)

    elif act_type in ["ATTACK_PLAYER", "ATTACK_CREATURE", "BLOCK", "BLOCK_CREATURE", "BREAK_SHIELD",
                      "RESOLVE_BATTLE", "RESOLVE_EFFECT", "USE_SHIELD_TRIGGER", "RESOLVE_PLAY"]:
        _handle_engine_execution(act_type, act_data, cmd)

    elif act_type == "PASS":
        cmd['type'] = "PASS"

    elif act_type in ["LOOK_TO_BUFFER", "REVEAL_TO_BUFFER", "SELECT_FROM_BUFFER", "PLAY_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE", "SUMMON_TOKEN"]:
        _handle_buffer_ops(act_type, act_data, cmd, src, dest)

    else:
        # Fallback / Special Legacy Keyword
        if act_type in ("NONE", "") and act_data.get('str_val'):
             cmd['type'] = 'ADD_KEYWORD'
             cmd['mutation_kind'] = str(act_data.get('str_val'))
             cmd['amount'] = act_data.get('value1', 1)
             _transfer_targeting(act_data, cmd)
        else:
            cmd['type'] = "NONE"
            cmd['legacy_warning'] = True
            # legacy_original_type is already set at init, but re-asserting logic here
            cmd['legacy_original_type'] = act_type
            cmd['str_param'] = f"Legacy: {act_type}"
            _transfer_targeting(act_data, cmd)

    # Annotate command with legacy alias info for short-term compatibility
    try:
        add_aliases_to_command(cmd, act_type)
    except Exception:
        pass

    _finalize_command(cmd, act_data)
    _validate_command_type(cmd)
    return cmd

# --- Sub-handlers ---

def _create_error_command(orig_val: str, msg: str) -> Dict[str, Any]:
    return {
        "type": "NONE",
        "uid": str(uuid.uuid4()),
        "legacy_warning": True,
        "legacy_original_value": orig_val,
        "str_param": msg
    }


def _handle_replace_card_move(act: Dict[str, Any], cmd: Dict[str, Any], original_zone: Optional[str], dest: Optional[str]) -> None:
    """
    Handle replacement-style card movement where an incoming zone is replaced
    with another destination (e.g., "墓地に置くかわりに山札の下に置く").

    The command remains a distinct type (REPLACE_CARD_MOVE) for text generation,
    but execution can fallback to TRANSITION semantics via EngineCompat.
    """
    cmd['type'] = 'REPLACE_CARD_MOVE'
    cmd['reason'] = 'REPLACE_CARD_MOVE'

    # Preserve the original intended destination separately for clarity
    if original_zone:
        cmd['original_to_zone'] = original_zone
        # Also expose as from_zone so existing UI/filters can reuse it
        cmd.setdefault('from_zone', original_zone)

    # Replacement destination defaults to deck bottom when omitted
    cmd['to_zone'] = dest or 'DECK_BOTTOM'

    # If an explicit current zone is provided, keep it; otherwise, rely on instance targeting
    if 'source_zone' in act:
        cmd['current_zone'] = act.get('source_zone')

    # Preserve explicit targeting hints
    if 'instance_id' in act:
        cmd['instance_id'] = act['instance_id']
    if 'source_instance_id' in act and 'instance_id' not in cmd:
        cmd['instance_id'] = act['source_instance_id']
    if 'target_instance' in act and 'instance_id' not in cmd:
        cmd['instance_id'] = act['target_instance']

    _transfer_common_move_fields(act, cmd)
    if 'amount' in cmd and cmd['amount'] == 0:
        cmd['amount'] = AMOUNT_ALL

def _handle_move_card(act: Dict[str, Any], cmd: Dict[str, Any], src: Optional[str], dest: Optional[str]) -> None:
    # Map generic MOVE_CARD to a more specific command when destination implies it.
    # Default to TRANSITION when no clearer mapping exists.
    # Default mapping for a generic MOVE_CARD should be a TRANSITION
    # Specific move types should be preserved if the incoming action explicitly names them.
    mapped_type = 'TRANSITION'

    # Try to infer specific command from destination if generic MOVE_CARD
    # NOTE: Avoid implicitly mapping MOVE_CARD->MANA_CHARGE based solely on destination.
    # Keep default as TRANSITION unless the incoming action explicitly names a specific move.
    if dest == "GRAVEYARD" and src == "BATTLE":
        mapped_type = "DESTROY"
    elif dest == "GRAVEYARD" and src == "HAND":
        mapped_type = "DISCARD"
    elif dest == "HAND" and src in ["BATTLE", "MANA", "SHIELD"]:
        mapped_type = "RETURN_TO_HAND"

    act_type = str(act.get('type', '')).upper()
    if act_type in ("MANA_CHARGE", "RETURN_TO_HAND", "DESTROY", "DISCARD"):
        mapped_type = act_type

    cmd['type'] = mapped_type
    if dest and 'to_zone' not in cmd:
        cmd['to_zone'] = dest
    if src and 'from_zone' not in cmd:
        cmd['from_zone'] = src

    _transfer_common_move_fields(act, cmd)
    if 'amount' in cmd and cmd['amount'] == 0:
        cmd['amount'] = AMOUNT_ALL

def _handle_specific_moves(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any], src: Optional[str]) -> None:
    """
    Handle move actions that imply specific source/destination zones.

    Applies default zone transitions when not explicitly provided:
    - MANA_CHARGE: HAND -> MANA
    - ADD_MANA: DECK -> MANA
    - ADD_SHIELD: DECK -> SHIELD
    - SEARCH_DECK_BOTTOM: DECK -> DECK_BOTTOM
    - DESTROY: [Any] -> GRAVEYARD
    """
    # Preserve the specific action type for compatibility with tests
    cmd['type'] = act_type
    # Also record original semantic intent as a reason for compatibility
    cmd['reason'] = act_type

    # Special-case amount/flags retained below where needed
    if act_type == "SHIELD_BURN":
         cmd['amount'] = act.get('amount') or act.get('value1', 1)

    # NOTE: These strings should match dm_ai_module.Zone enum names if possible
    # to be picked up by compat.py correctly.
    if act_type in ["SEND_TO_MANA", "MANA_CHARGE", "ADD_MANA"]:
        cmd['to_zone'] = "MANA"
        if src:
            cmd['from_zone'] = src
        elif act_type == "MANA_CHARGE":
            cmd['from_zone'] = "HAND"
        elif act_type == "ADD_MANA":
            cmd['from_zone'] = "DECK"
    elif act_type in ["SEND_TO_DECK_BOTTOM", "SEARCH_DECK_BOTTOM"]:
        # SEARCH_DECK_BOTTOM also implies moving card to deck bottom (often from deck top or hand)
        cmd['to_zone'] = "DECK_BOTTOM"
        if not src and act_type == "SEARCH_DECK_BOTTOM": cmd['from_zone'] = "DECK"
    elif act_type == "ADD_SHIELD":
        cmd['to_zone'] = "SHIELD"
        if not src: cmd['from_zone'] = "DECK"
    elif act_type == "DESTROY":
        cmd['to_zone'] = "GRAVEYARD"
        if src: cmd['from_zone'] = src
    elif act_type == 'RETURN_TO_HAND':
        cmd['to_zone'] = "HAND"
        if src: cmd['from_zone'] = src
    elif act_type == 'DISCARD':
        cmd['to_zone'] = "GRAVEYARD"
        cmd['from_zone'] = "HAND"
    elif act_type == "MOVE_TO_UNDER_CARD":
        # Consistently use TRANSITION for movement to UNDER_CARD.
        # Legacy mapper used ATTACH, but unification prefers TRANSITION with 'to_zone'.
        # Note: The engine must support 'UNDER_CARD' as a destination zone or handle base_target.
        cmd['type'] = "TRANSITION"
        cmd['to_zone'] = "UNDER_CARD"
        if src: cmd['from_zone'] = src
        # Preserve base_target if present
        if 'base_target' in act:
            cmd['base_target'] = act['base_target']

    _transfer_common_move_fields(act, cmd)

    # Map explicit 0 amount to ALL (255) for targeting moves
    if act_type in ["DESTROY", "DISCARD", "RETURN_TO_HAND", "SEND_TO_MANA"]:
        if 'amount' in cmd and cmd['amount'] == 0:
            cmd['amount'] = AMOUNT_ALL

def _handle_modifiers(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any]) -> None:
    val = act.get('str_param') or act.get('str_val', '')
    if act_type == "COST_REDUCTION" or val == "COST":
        cmd['type'] = "MUTATE"
        cmd['mutation_kind'] = "COST"
        cmd['amount'] = act.get('amount') or act.get('value1', 0)
    elif act_type == "GRANT_KEYWORD":
        cmd['type'] = "ADD_KEYWORD"
        cmd['mutation_kind'] = act.get('str_param') or act.get('str_val', '')
        cmd['amount'] = act.get('amount') or act.get('value1', 1)
    else:
        cmd['type'] = "MUTATE"
        cmd['str_param'] = val
        if 'amount' in act: cmd['amount'] = act['amount']
        elif 'value1' in act: cmd['amount'] = act['value1']
    _transfer_targeting(act, cmd)

def _handle_mutate(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any]) -> None:
    sval = str(act.get('str_param') or act.get('str_val') or '').upper()

    if sval in ("TAP", "UNTAP"):
        cmd['type'] = sval
        if 'amount' in act: cmd['amount'] = act['amount']
        elif 'value1' in act: cmd['amount'] = act['value1']
    elif sval == "SHIELD_BURN":
        cmd['type'] = "SHIELD_BURN"
        if 'amount' in act: cmd['amount'] = act['amount']
        elif 'value1' in act: cmd['amount'] = act['value1']
    elif sval in ("SET_POWER", "POWER_SET"):
        cmd['type'] = 'MUTATE'
        cmd['mutation_kind'] = 'POWER_SET'
        if 'amount' in act: cmd['amount'] = act['amount']
        elif 'value1' in act: cmd['amount'] = act['value1']
    # Consolidate POWER_MOD into generic MUTATE with mutation_kind
    # Check this AFTER specific power-set operations
    elif act_type in ["POWER_MOD", "MODIFY_POWER"] or 'POWER' in sval:
        cmd['type'] = 'MUTATE'
        cmd['mutation_kind'] = 'POWER_MOD'
        if 'amount' in act: cmd['amount'] = act['amount']
        elif 'value1' in act: cmd['amount'] = act['value1']
        elif 'value2' in act: cmd['amount'] = act['value2']
    elif 'HEAL' in sval or 'RECOVER' in sval:
        cmd['type'] = 'MUTATE'
        cmd['mutation_kind'] = 'HEAL'
        if 'amount' in act: cmd['amount'] = act['amount']
        elif 'value1' in act: cmd['amount'] = act['value1']
    elif 'REMOVE_KEYWORD' in sval:
         cmd['type'] = 'MUTATE'
         cmd['mutation_kind'] = 'REMOVE_KEYWORD'
    else:
        cmd['type'] = "MUTATE"
        cmd['str_param'] = act.get('str_param') or act.get('str_val')
        if 'amount' in act: cmd['amount'] = act['amount']
        elif 'value1' in act: cmd['amount'] = act['value1']
    _transfer_targeting(act, cmd)

    # Map explicit 0 amount to ALL for TAP/UNTAP
    if cmd.get('type') in ["TAP", "UNTAP"]:
        if 'amount' in cmd and cmd['amount'] == 0:
            cmd['amount'] = AMOUNT_ALL

def _handle_selection(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any]) -> None:
    if act_type == "SELECT_OPTION":
        # Always map to CHOICE; enum exposure differences are handled in validation.
        cmd['type'] = "CHOICE"

        cmd['amount'] = act.get('amount') or act.get('value1', 1)
        if act.get('value2', 0) == 1:
            flags = cast(List[Any], cmd.setdefault('flags', []))
            flags.append("ALLOW_DUPLICATES")
    
    elif act_type == "SELECT_TARGET":
        cmd['type'] = "QUERY"
        cmd['str_param'] = "SELECT_TARGET"
        if cmd.get('target_group') == 'NONE' and 'target_group' not in act:
             cmd['target_group'] = 'TARGET_SELECT'
    elif act_type == "SELECT_NUMBER":
        # Preserve legacy SELECT_NUMBER command while providing numeric bounds
        cmd['type'] = "SELECT_NUMBER"
        if 'max' in act: cmd['max'] = int(act.get('max') or 0)
        elif 'value1' in act:
            cmd['max'] = int(act.get('value1') or 0)

        if 'min' in act: cmd['min'] = int(act.get('min') or 0)
        elif 'value2' in act:
            cmd['min'] = int(act.get('value2') or 0)
        # default amount is 1
        cmd['amount'] = act.get('amount', 1)
        _transfer_targeting(act, cmd)
    else:
        _transfer_targeting(act, cmd)

def _handle_complex(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any], dest: Optional[str]) -> None:
    # Map complex/deck effects while preserving legacy 'type' for tests and compat.
    if act_type == "SEARCH_DECK":
        cmd['type'] = "SEARCH_DECK"
        # Unified representation
        cmd['unified_type'] = 'SEARCH'
        cmd['search_type'] = 'SEARCH_DECK'
        cmd['amount'] = act.get('value1', 1)
        if 'filter' in act:
            cmd['target_filter'] = copy.deepcopy(act['filter'])
    elif act_type == "SHUFFLE_DECK":
        cmd['type'] = "SHUFFLE_DECK"
    elif act_type == "REVEAL_CARDS":
        cmd['type'] = "REVEAL_CARDS"
        cmd['amount'] = act.get('value1', 1)
    elif act_type == "LOOK_AND_ADD":
        cmd['type'] = "LOOK_AND_ADD"
        cmd['unified_type'] = 'SEARCH'
        cmd['search_type'] = 'LOOK_AND_ADD'
        if 'value1' in act: cmd['look_count'] = int(act['value1'])
        elif 'filter' in act and 'count' in act['filter']: cmd['look_count'] = act['filter']['count']
        if 'value2' in act: cmd['add_count'] = int(act['value2'])
        elif 'filter' in act and 'select' in act['filter']: cmd['add_count'] = act['filter']['select']
        if 'rest_zone' in act: cmd['rest_zone'] = act['rest_zone']
        elif 'destination_zone' in act: cmd['rest_zone'] = act['destination_zone']
    elif act_type == "MEKRAID":
        cmd['type'] = "MEKRAID"
        cmd['unified_type'] = 'SEARCH'
        cmd['search_type'] = 'MEKRAID'
        cmd['look_count'] = int(act.get('look_count') or act.get('value2') or 3)
        cmd['max_cost'] = act.get('value1', 0)
        cmd['select_count'] = 1
        cmd['play_for_free'] = True
        cmd['rest_zone'] = act.get('rest_zone') or 'DECK_BOTTOM'
        # If minimal parameters are missing, mark as legacy-warning for manual review
        if 'value1' not in act and 'look_count' not in act and 'filter' not in act:
            cmd['legacy_warning'] = True
    elif act_type == "REVOLUTION_CHANGE":
        cmd['type'] = "MUTATE"
        cmd['mutation_kind'] = 'REVOLUTION_CHANGE'
        if 'value1' in act: cmd['amount'] = act['value1']
        if 'str_val' in act: cmd['str_param'] = act['str_val']
    _transfer_targeting(act, cmd)

def _handle_play_flow(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any], src: Optional[str], dest: Optional[str]) -> None:
    if act_type == "PLAY_CARD" or act_type == "DECLARE_PLAY":
        # Direct play from hand
        cmd['type'] = "PLAY_FROM_ZONE"
        cmd['from_zone'] = src or "HAND"
        cmd['to_zone'] = dest or "BATTLE"
        # Unified hint for downstream detection
        cmd['unified_type'] = 'PLAY'
        if 'value1' in act: cmd['amount'] = act['value1']
    elif act_type == "PLAY_FROM_ZONE":
        # Consolidate play commands to a single PLAY command with from_zone
        cmd['type'] = "PLAY_FROM_ZONE"
        if src: cmd['from_zone'] = src
        cmd['to_zone'] = dest or 'BATTLE'
        if 'value1' in act: cmd['max_cost'] = act['value1']
        cmd['str_param'] = "PLAY_FROM_ZONE_HINT"
        cmd['unified_type'] = 'PLAY'
        # propagate explicit play_for_free flag if present
        if act.get('play_for_free') or act.get('play_free'):
            play_flags = cast(List[Any], cmd.setdefault('play_flags', []))
            play_flags.append('PLAY_FOR_FREE')
        # propagate put-into-play semantic if action explicitly requests it
        if act.get('put_into_play') or act.get('force_put'):
            play_flags = cast(List[Any], cmd.setdefault('play_flags', []))
            play_flags.append('PUT_IN_PLAY')
    elif act_type == "FRIEND_BURST":
        cmd['type'] = "FRIEND_BURST"
        cmd['str_val'] = act.get('str_val')
        if 'value1' in act: cmd['value1'] = act['value1']
        cmd['unified_type'] = 'PLAY'
    elif act_type == "REGISTER_DELAYED_EFFECT":
        cmd['type'] = "REGISTER_DELAYED_EFFECT"
        cmd['str_val'] = act.get('str_val')
        if 'value1' in act: cmd['value1'] = act['value1']
    elif act_type == "CAST_SPELL":
        cmd['type'] = "CAST_SPELL"
        if 'str_val' in act: cmd['str_val'] = act['str_val']
        cmd['unified_type'] = 'PLAY'

    _transfer_targeting(act, cmd)

def _handle_engine_execution(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any]) -> None:
    if act_type == "ATTACK_PLAYER":
        cmd['type'] = "ATTACK_PLAYER"
        cmd['instance_id'] = _get_any(act, ['source_instance', 'source_instance_id', 'attacker_id'])
        cmd['target_player'] = act.get('target_player')
    elif act_type == "ATTACK_CREATURE":
        cmd['type'] = "ATTACK_CREATURE"
        cmd['instance_id'] = _get_any(act, ['source_instance', 'source_instance_id', 'attacker_id'])
        cmd['target_instance'] = _get_any(act, ['target_instance', 'target_instance_id', 'target_id'])
    elif act_type == "BLOCK" or act_type == "BLOCK_CREATURE":
        cmd['type'] = "FLOW"
        cmd['flow_type'] = "BLOCK"
        cmd['instance_id'] = _get_any(act, ['blocker_id', 'source_instance_id'])
        cmd['target_instance'] = _get_any(act, ['attacker_id', 'target_instance_id', 'target_id'])
    elif act_type == "BREAK_SHIELD":
        cmd['type'] = "BREAK_SHIELD"
        cmd['amount'] = act.get('value1', 1)
        if 'creature_id' in act: cmd['instance_id'] = act['creature_id']
        _transfer_targeting(act, cmd)
    elif act_type == "RESOLVE_BATTLE":
        cmd['type'] = "RESOLVE_BATTLE"
        if 'winner_id' in act: cmd['winner_instance'] = act['winner_id']
    elif act_type == "RESOLVE_EFFECT":
        cmd['type'] = "RESOLVE_EFFECT"
        # C++ uses slot_index to store effect index for RESOLVE_EFFECT actions
        if 'slot_index' in act:
            cmd['effect_index'] = act['slot_index']
        if 'effect_id' in act:
            cmd['effect_id'] = act['effect_id']
    elif act_type == "USE_SHIELD_TRIGGER":
        cmd['type'] = "USE_SHIELD_TRIGGER"
        cmd['instance_id'] = _get_any(act, ['card_id', 'source_instance_id'])
    elif act_type == "RESOLVE_PLAY":
        cmd['type'] = "RESOLVE_PLAY"
        cmd['instance_id'] = _get_any(act, ['card_id', 'source_instance_id'])

def _handle_buffer_ops(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any], src: Optional[str], dest: Optional[str]) -> None:
    if act_type == "LOOK_TO_BUFFER":
        cmd['type'] = 'LOOK_TO_BUFFER'
        cmd['look_count'] = act.get('value1', 1)
        if src:
            cmd['from_zone'] = src
    elif act_type == "REVEAL_TO_BUFFER":
        cmd['type'] = 'REVEAL_TO_BUFFER'
        cmd['look_count'] = act.get('value1', 1)
        if src:
            cmd['from_zone'] = src
    elif act_type == "SELECT_FROM_BUFFER":
        cmd['type'] = 'SELECT_FROM_BUFFER'
        cmd['amount'] = act.get('value1', 1)
        if act.get('value2', 0) == 1:
            flags = cast(List[Any], cmd.setdefault('flags', []))
            flags.append('ALLOW_DUPLICATES')
    elif act_type == "PLAY_FROM_BUFFER":
        # Preserve legacy type while indicating unified PLAY semantics
        cmd['type'] = 'PLAY_FROM_BUFFER'
        cmd['unified_type'] = 'PLAY'
        cmd['from_zone'] = 'BUFFER'
        # Use normalized destination if available
        cmd['to_zone'] = dest or 'BATTLE'
        if 'value1' in act: cmd['max_cost'] = act['value1']
    elif act_type == "MOVE_BUFFER_TO_ZONE":
        cmd['type'] = 'TRANSITION'
        cmd['from_zone'] = 'BUFFER'
        cmd['to_zone'] = dest or 'HAND'
        if 'value1' in act: cmd['amount'] = act['value1']
    elif act_type == "SUMMON_TOKEN":
        cmd['type'] = "SUMMON_TOKEN"
        if 'value1' in act: cmd['amount'] = act['value1']
        if 'str_val' in act: cmd['token_id'] = act['str_val']

    _transfer_targeting(act, cmd)

# Unified Entry Point Alias
action_to_command = map_action
