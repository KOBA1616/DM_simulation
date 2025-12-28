# -*- coding: utf-8 -*-
import uuid
import copy
from typing import Any, Dict, List, Optional

# Try to import dm_ai_module to get enums, otherwise define mocks/None.
# Note: In some environments the compiled extension may be missing or fail to load.
try:
    import dm_ai_module  # type: ignore
except Exception:
    dm_ai_module = None  # type: ignore

_CommandType = getattr(dm_ai_module, 'CommandType', None) if dm_ai_module is not None else None
_Zone = getattr(dm_ai_module, 'Zone', None) if dm_ai_module is not None else None

def normalize_action_zone_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures action dictionary has consistent zone keys."""
    if not isinstance(data, dict): return data
    new_data = data.copy()
    if 'source_zone' not in new_data and 'from_zone' in new_data: new_data['source_zone'] = new_data['from_zone']
    if 'destination_zone' not in new_data and 'to_zone' in new_data: new_data['destination_zone'] = new_data['to_zone']
    return new_data

def _get_zone(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d: return d[k]
    return None

def _transfer_targeting(act: Dict[str, Any], cmd: Dict[str, Any]):
    scope = act.get('scope', 'NONE')
    if scope == 'NONE' and 'filter' in act:
         scope = 'TARGET_SELECT'
    cmd['target_group'] = scope
    if 'filter' in act:
        cmd['target_filter'] = copy.deepcopy(act['filter'])
    if act.get('optional', False):
        cmd.setdefault('flags', []).append('OPTIONAL')

def _transfer_common_move_fields(act: Dict[str, Any], cmd: Dict[str, Any]):
    _transfer_targeting(act, cmd)
    if 'filter' in act and isinstance(act['filter'], dict) and 'count' in act['filter']:
         cmd['amount'] = act['filter']['count']
    elif 'value1' in act:
         cmd['amount'] = act['value1']

def _finalize_command(cmd: Dict[str, Any], act: Dict[str, Any]):
    if 'uid' not in cmd:
        cmd['uid'] = str(uuid.uuid4())

    # Ensure amount/flags exist
    if 'amount' in act and 'amount' not in cmd:
        try:
            cmd['amount'] = int(act.get('amount') or act.get('value1') or 0)
        except Exception:
             pass # Ignore if not convertible

def map_action(action_data: Any) -> Dict[str, Any]:
    """
    Pure function to convert a legacy Action dictionary/object to a Command dictionary.
    """
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
        "legacy_warning": False
    }

    # Common Fields
    for key in ['input_value_key', 'output_value_key', 'uid']:
        if key in act_data:
            cmd[key] = act_data[key]

    # Recursion (Options)
    if 'options' in act_data and isinstance(act_data['options'], list):
        cmd['options'] = []
        for opt in act_data['options']:
            if isinstance(opt, list):
                cmd['options'].append([map_action(sub) for sub in opt])
            else:
                cmd['options'].append([map_action(opt)])

    src = _get_zone(act_data, ['source_zone', 'from_zone', 'origin_zone'])
    dest = _get_zone(act_data, ['destination_zone', 'to_zone', 'dest_zone'])

    # Phase 4.2: Normalize Zone Names to Enum
    # Mappings from Legacy/Common strings to dm_ai_module.Zone Enum names
    zone_map = {
        "MANA_ZONE": "MANA",
        "BATTLE_ZONE": "BATTLE",
        "SHIELD_ZONE": "SHIELD",
        "GRAVEYARD": "GRAVEYARD",
        "HAND": "HAND",
        "DECK": "DECK"
    }

    if src and src in zone_map: src = zone_map[src]
    if dest and dest in zone_map: dest = zone_map[dest]

    # --- Logic Mapping ---

    if act_type == "MOVE_CARD":
        _handle_move_card(act_data, cmd, src, dest)

    elif act_type in ["DESTROY", "DISCARD", "MANA_CHARGE", "RETURN_TO_HAND",
                      "SEND_TO_MANA", "SEND_TO_DECK_BOTTOM", "ADD_SHIELD", "SHIELD_BURN"]:
        _handle_specific_moves(act_type, act_data, cmd, src)

    elif act_type == "DRAW_CARD":
        # DECISION: We unify DRAW into TRANSITION(DECK->HAND)
        cmd['type'] = "TRANSITION"
        cmd['from_zone'] = src or 'DECK'
        cmd['to_zone'] = dest or 'HAND'
        _transfer_common_move_fields(act_data, cmd)

    elif act_type in ["TAP", "UNTAP"]:
        cmd['type'] = act_type
        _transfer_targeting(act_data, cmd)

    elif act_type in ["COUNT_CARDS", "MEASURE_COUNT", "GET_GAME_STAT"]:
        cmd['type'] = "QUERY"
        cmd['str_param'] = act_data.get('str_val', 'CARDS_MATCHING_FILTER' if 'COUNT' in act_type else '')
        _transfer_targeting(act_data, cmd)

    elif act_type in ["APPLY_MODIFIER", "COST_REDUCTION", "GRANT_KEYWORD"]:
        _handle_modifiers(act_type, act_data, cmd)

    elif act_type in ["MUTATE", "POWER_MOD"]:
        _handle_mutate(act_type, act_data, cmd)

    elif act_type in ["SELECT_OPTION", "SELECT_NUMBER", "SELECT_TARGET"]:
        _handle_selection(act_type, act_data, cmd)

    elif act_type in ["SEARCH_DECK", "SHUFFLE_DECK", "REVEAL_CARDS", "LOOK_AND_ADD", "MEKRAID", "REVOLUTION_CHANGE"]:
        _handle_complex(act_type, act_data, cmd, dest)

    elif act_type in ["PLAY_FROM_ZONE", "FRIEND_BURST", "REGISTER_DELAYED_EFFECT", "CAST_SPELL"]:
        _handle_play_flow(act_type, act_data, cmd, src, dest)

    elif act_type in ["ATTACK_PLAYER", "ATTACK_CREATURE", "BLOCK", "BREAK_SHIELD",
                      "RESOLVE_BATTLE", "RESOLVE_EFFECT", "USE_SHIELD_TRIGGER", "RESOLVE_PLAY"]:
        _handle_engine_execution(act_type, act_data, cmd)

    elif act_type in ["LOOK_TO_BUFFER", "SELECT_FROM_BUFFER", "PLAY_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE", "SUMMON_TOKEN"]:
        _handle_buffer_ops(act_type, act_data, cmd, dest)

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
            cmd['legacy_original_type'] = act_type
            cmd['str_param'] = f"Legacy: {act_type}"
            _transfer_targeting(act_data, cmd)

    _finalize_command(cmd, act_data)
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

def _handle_move_card(act, cmd, src, dest):
    # Phase 4.2 Normalization: Prefer TRANSITION for all standard moves
    cmd['type'] = "TRANSITION"
    if dest and 'to_zone' not in cmd: cmd['to_zone'] = dest
    if src and 'from_zone' not in cmd: cmd['from_zone'] = src

    _transfer_common_move_fields(act, cmd)

def _handle_specific_moves(act_type, act, cmd, src):
    if act_type == "SHIELD_BURN":
         cmd['type'] = "SHIELD_BURN"
         cmd['amount'] = act.get('value1', 1)
    else:
        cmd['type'] = "TRANSITION"

    # NOTE: These strings should match dm_ai_module.Zone enum names if possible
    # to be picked up by compat.py correctly.
    if act_type in ["SEND_TO_MANA", "MANA_CHARGE"]:
        cmd['to_zone'] = "MANA"
        if not src and act_type == "MANA_CHARGE": cmd['from_zone'] = "DECK"
        if src: cmd['from_zone'] = src
    elif act_type == "SEND_TO_DECK_BOTTOM":
        cmd['to_zone'] = "DECK_BOTTOM" # Special case, likely not an enum. compat.py needs to handle it or fallback.
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

    _transfer_common_move_fields(act, cmd)

def _handle_modifiers(act_type, act, cmd):
    val = act.get('str_val', '')
    if act_type == "COST_REDUCTION" or val == "COST":
        cmd['type'] = "MUTATE"
        cmd['mutation_kind'] = "COST"
        cmd['amount'] = act.get('value1', 0)
    elif act_type == "GRANT_KEYWORD":
        cmd['type'] = "ADD_KEYWORD"
        cmd['mutation_kind'] = act.get('str_val', '')
        cmd['amount'] = act.get('value1', 1)
    else:
        cmd['type'] = "MUTATE"
        cmd['str_param'] = val
        if 'value1' in act: cmd['amount'] = act['value1'] # Explicitly transfer value1 if present
    _transfer_targeting(act, cmd)

def _handle_mutate(act_type, act, cmd):
    sval = str(act.get('str_val') or '').upper()
    if act_type == "POWER_MOD" or 'POWER' in sval:
        cmd['type'] = 'POWER_MOD'
        cmd['mutation_kind'] = 'POWER_MOD'
        if 'value1' in act: cmd['amount'] = act['value1']
        elif 'value2' in act: cmd['amount'] = act['value2']
    elif sval in ("TAP", "UNTAP"):
        cmd['type'] = sval
    elif sval == "SHIELD_BURN":
        cmd['type'] = "SHIELD_BURN"
        if 'value1' in act: cmd['amount'] = act['value1']
    elif sval in ("SET_POWER", "POWER_SET"):
        cmd['type'] = 'MUTATE'
        cmd['mutation_kind'] = 'POWER_SET'
        if 'value1' in act: cmd['amount'] = act['value1']
    elif 'HEAL' in sval or 'RECOVER' in sval:
        cmd['type'] = 'MUTATE'
        cmd['mutation_kind'] = 'HEAL'
        if 'value1' in act: cmd['amount'] = act['value1']
    elif 'REMOVE_KEYWORD' in sval:
         cmd['type'] = 'MUTATE'
         cmd['mutation_kind'] = 'REMOVE_KEYWORD'
    else:
        cmd['type'] = "MUTATE"
        cmd['str_param'] = act.get('str_val')
        if 'value1' in act: cmd['amount'] = act['value1']
    _transfer_targeting(act, cmd)

def _handle_selection(act_type, act, cmd):
    if act_type == "SELECT_OPTION":
        cmd['type'] = "CHOICE"
        cmd['amount'] = act.get('value1', 1)
        if act.get('value2', 0) == 1:
            cmd.setdefault('flags', []).append("ALLOW_DUPLICATES")
    elif act_type == "SELECT_NUMBER":
        cmd['type'] = "SELECT_NUMBER"
        if 'value1' in act:
            cmd['max'] = int(act.get('value1') or 0)
    elif act_type == "SELECT_TARGET":
        cmd['type'] = "QUERY"
        cmd['str_param'] = "SELECT_TARGET"
        if cmd.get('target_group') == 'NONE' and 'target_group' not in act:
             cmd['target_group'] = 'TARGET_SELECT'
    _transfer_targeting(act, cmd)

def _handle_complex(act_type, act, cmd, dest):
    if act_type == "SEARCH_DECK":
        cmd['type'] = "SEARCH_DECK"
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
        if 'value1' in act: cmd['look_count'] = int(act['value1'])
        elif 'filter' in act and 'count' in act['filter']: cmd['look_count'] = act['filter']['count']
        if 'value2' in act: cmd['add_count'] = int(act['value2'])
        elif 'filter' in act and 'select' in act['filter']: cmd['add_count'] = act['filter']['select']
        if 'rest_zone' in act: cmd['rest_zone'] = act['rest_zone']
        elif 'destination_zone' in act: cmd['rest_zone'] = act['destination_zone']
    elif act_type == "MEKRAID":
        cmd['type'] = "MEKRAID"
        cmd['look_count'] = int(act.get('look_count') or act.get('value2') or 3)
        cmd['max_cost'] = act.get('value1', 0)
        cmd['select_count'] = 1
        cmd['play_for_free'] = True
        cmd['rest_zone'] = act.get('rest_zone') or 'DECK_BOTTOM'
    elif act_type == "REVOLUTION_CHANGE":
        cmd['type'] = "MUTATE"
        cmd['mutation_kind'] = 'REVOLUTION_CHANGE'
        if 'value1' in act: cmd['amount'] = act['value1']
        if 'str_val' in act: cmd['str_param'] = act['str_val']
    _transfer_targeting(act, cmd)

def _handle_play_flow(act_type, act, cmd, src, dest):
    if act_type == "PLAY_FROM_ZONE":
        cmd['type'] = "PLAY_FROM_ZONE"
        if src: cmd['from_zone'] = src
        cmd['to_zone'] = dest or 'BATTLE'
        if 'value1' in act: cmd['max_cost'] = act['value1']
        cmd['str_param'] = "PLAY_FROM_ZONE_HINT"
    elif act_type == "FRIEND_BURST":
        cmd['type'] = "FRIEND_BURST"
        cmd['str_val'] = act.get('str_val')
        if 'value1' in act: cmd['value1'] = act['value1']
    elif act_type == "REGISTER_DELAYED_EFFECT":
        cmd['type'] = "REGISTER_DELAYED_EFFECT"
        cmd['str_val'] = act.get('str_val')
        if 'value1' in act: cmd['value1'] = act['value1']
    elif act_type == "CAST_SPELL":
        cmd['type'] = "CAST_SPELL"
        if 'str_val' in act: cmd['str_val'] = act['str_val']

    _transfer_targeting(act, cmd)

def _handle_engine_execution(act_type, act, cmd):
    if act_type == "ATTACK_PLAYER":
        cmd['type'] = "ATTACK_PLAYER"
        cmd['instance_id'] = act.get('source_instance') or act.get('source_instance_id') or act.get('attacker_id')
        cmd['target_player'] = act.get('target_player')
    elif act_type == "ATTACK_CREATURE":
        cmd['type'] = "ATTACK_CREATURE"
        cmd['instance_id'] = act.get('source_instance') or act.get('source_instance_id') or act.get('attacker_id')
        cmd['target_instance'] = act.get('target_instance') or act.get('target_instance_id') or act.get('target_id')
    elif act_type == "BLOCK":
        cmd['type'] = "FLOW"
        cmd['flow_type'] = "BLOCK"
        cmd['instance_id'] = act.get('blocker_id') or act.get('source_instance_id')
        cmd['target_instance'] = act.get('attacker_id') or act.get('target_instance_id')
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
        if 'effect_id' in act: cmd['effect_id'] = act['effect_id']
    elif act_type == "USE_SHIELD_TRIGGER":
        cmd['type'] = "USE_SHIELD_TRIGGER"
        cmd['instance_id'] = act.get('card_id') or act.get('source_instance_id')
    elif act_type == "RESOLVE_PLAY":
        cmd['type'] = "RESOLVE_PLAY"
        cmd['instance_id'] = act.get('card_id') or act.get('source_instance_id')

def _handle_buffer_ops(act_type, act, cmd, dest):
    if act_type == "LOOK_TO_BUFFER":
        cmd['type'] = 'LOOK_TO_BUFFER'
        cmd['look_count'] = act.get('value1', 1)
    elif act_type == "SELECT_FROM_BUFFER":
        cmd['type'] = 'SELECT_FROM_BUFFER'
        cmd['amount'] = act.get('value1', 1)
        if act.get('value2', 0) == 1: cmd.setdefault('flags', []).append('ALLOW_DUPLICATES')
    elif act_type == "PLAY_FROM_BUFFER":
        cmd['type'] = 'PLAY_FROM_BUFFER'
        cmd['from_zone'] = 'BUFFER'
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
