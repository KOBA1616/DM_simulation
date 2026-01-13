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
    Handles C++ pybind11 objects and legacy dicts.
    """
    if not card_data:
        return ["COLORLESS"]

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
                return card_db[card_id].name  # type: ignore
    except Exception:
        pass

    return f"Inst_{instance_id}"

def convert_card_data_to_dict(card_data: Any) -> Dict[str, Any]:
    """
    Converts a CardDefinition/CardData C++ object to a Python dictionary
    compatible with CardTextGenerator.
    """
    if not card_data:
        return {}

    # If it's already a dict, return it
    if isinstance(card_data, dict):
        return card_data

    # Helper to safely get attributes or defaults
    def get(obj, attr, default=None):
        return getattr(obj, attr, default)

    # Convert simple fields
    data = {
        'id': get(card_data, 'id', -1),
        'name': get(card_data, 'name', 'Unknown'),
        'cost': get(card_data, 'cost', 0),
        'power': get(card_data, 'power', 0),
        'civilizations': get_card_civilizations(card_data),
        'races': list(get(card_data, 'races', [])),
    }

    # Type
    raw_type = get(card_data, 'type')
    if hasattr(raw_type, 'name'):
        data['type'] = raw_type.name
    else:
        data['type'] = str(raw_type).split('.')[-1] if raw_type else "CREATURE"

    # Keywords
    keywords_obj = get(card_data, 'keywords')
    keywords = {}
    if keywords_obj:
        # Manually map known keywords from CardKeywords struct
        # This list must match CardKeywords property bindings
        kw_list = [
            "g_zero", "revolution_change", "mach_fighter", "speed_attacker", "blocker",
            "slayer", "double_breaker", "triple_breaker", "shield_trigger", "evolution",
            "neo", "g_neo", "cip", "at_attack", "destruction", "before_break_shield",
            "just_diver", "hyper_energy", "at_block", "at_start_of_turn", "at_end_of_turn",
            "g_strike", "world_breaker", "power_attacker", "shield_burn", "untap_in",
            "unblockable", "friend_burst", "ex_life", "mega_last_burst", "must_be_chosen"
        ]
        for kw in kw_list:
            if hasattr(keywords_obj, kw):
                keywords[kw] = getattr(keywords_obj, kw)
    data['keywords'] = keywords

    # Effects
    effects_objs = get(card_data, 'effects', [])
    data['effects'] = [_convert_effect(eff) for eff in effects_objs]

    # Revolution Change Condition
    rev_cond = get(card_data, 'revolution_change_condition')
    if rev_cond:
         data['revolution_change_condition'] = _convert_filter(rev_cond)

    # Spell Side (Twinpact)
    spell_side = get(card_data, 'spell_side')
    if spell_side:
        data['spell_side'] = convert_card_data_to_dict(spell_side)

    # Static Abilities
    static_abs = get(card_data, 'static_abilities', [])
    data['static_abilities'] = [_convert_modifier(mod) for mod in static_abs]

    # AI Importance Score
    data['ai_importance_score'] = get(card_data, 'ai_importance_score', 0)
    data['is_key_card'] = get(card_data, 'is_key_card', False)

    return data

def _convert_effect(eff_obj: Any) -> Dict[str, Any]:
    if not eff_obj: return {}

    # Trigger
    trig = getattr(eff_obj, 'trigger', None)
    trigger_str = "NONE"
    if hasattr(trig, 'name'):
        trigger_str = trig.name
    else:
        trigger_str = str(trig).split('.')[-1]

    # Commands
    cmds = getattr(eff_obj, 'commands', [])
    commands_list = [_convert_command(cmd) for cmd in cmds]

    # Actions (Legacy support if actions exist on C++ side but usually mapped to commands now)
    actions = getattr(eff_obj, 'actions', [])
    actions_list = [_convert_action(act) for act in actions]

    return {
        'trigger': trigger_str,
        'condition': _convert_condition(getattr(eff_obj, 'condition', None)),
        'commands': commands_list,
        'actions': actions_list
    }

def _convert_command(cmd_obj: Any) -> Dict[str, Any]:
    if not cmd_obj: return {}

    ctype = getattr(cmd_obj, 'type', None)
    type_str = "NONE"
    if hasattr(ctype, 'name'):
         type_str = ctype.name
    else:
         type_str = str(ctype).split('.')[-1]

    d = {
        'type': type_str,
        'target_group': _enum_to_str(getattr(cmd_obj, 'target_group', None)),
        'target_filter': _convert_filter(getattr(cmd_obj, 'target_filter', None)),
        'amount': getattr(cmd_obj, 'amount', 0),
        'str_param': getattr(cmd_obj, 'str_param', ""),
        'optional': getattr(cmd_obj, 'optional', False),
        'from_zone': _enum_to_str(getattr(cmd_obj, 'from_zone', None)),
        'to_zone': _enum_to_str(getattr(cmd_obj, 'to_zone', None)),
        'mutation_kind': _enum_to_str(getattr(cmd_obj, 'mutation_kind', None)), # Assuming string or enum
        'condition': _convert_condition(getattr(cmd_obj, 'condition', None)),
        'input_value_key': getattr(cmd_obj, 'input_value_key', ""),
        'input_value_usage': getattr(cmd_obj, 'input_value_usage', ""),
        'output_value_key': getattr(cmd_obj, 'output_value_key', "")
    }

    return d

def _convert_action(act_obj: Any) -> Dict[str, Any]:
    if not act_obj: return {}
    # Legacy ActionDef support
    atype = getattr(act_obj, 'type', None)
    type_str = "NONE"
    if hasattr(atype, 'name'):
        type_str = atype.name
    else:
        type_str = str(atype).split('.')[-1]

    return {
        'type': type_str,
        'value1': getattr(act_obj, 'value1', 0),
        'value2': getattr(act_obj, 'value2', 0),
        'str_val': getattr(act_obj, 'str_val', ""),
        'optional': getattr(act_obj, 'optional', False),
        'filter': _convert_filter(getattr(act_obj, 'filter', None)),
        'target_player': getattr(act_obj, 'target_player', 0),
        'source_zone': _enum_to_str(getattr(act_obj, 'source_zone', None)),
        'destination_zone': _enum_to_str(getattr(act_obj, 'destination_zone', None)),
        'scope': _enum_to_str(getattr(act_obj, 'scope', None)),
        'input_value_key': getattr(act_obj, 'input_value_key', ""),
        'input_value_usage': getattr(act_obj, 'input_value_usage', ""),
        'output_value_key': getattr(act_obj, 'output_value_key', "")
    }

def _convert_modifier(mod_obj: Any) -> Dict[str, Any]:
    if not mod_obj: return {}
    mtype = getattr(mod_obj, 'type', None)
    type_str = "NONE"
    if hasattr(mtype, 'name'):
        type_str = mtype.name
    else:
        type_str = str(mtype).split('.')[-1]

    return {
        'type': type_str,
        'value': getattr(mod_obj, 'value', 0),
        'str_val': getattr(mod_obj, 'str_val', ""),
        'condition': _convert_condition(getattr(mod_obj, 'condition', None)),
        'filter': _convert_filter(getattr(mod_obj, 'filter', None))
    }

def _convert_condition(cond_obj: Any) -> Dict[str, Any]:
    if not cond_obj: return {}
    ctype = getattr(cond_obj, 'type', None)
    # ConditionType is not strictly bound as Enum in some places, might be int or string
    # Assuming it behaves like other enums if it is one
    type_str = "NONE"
    if hasattr(ctype, 'name'):
         type_str = ctype.name
    elif hasattr(ctype, '__str__'):
         type_str = str(ctype).split('.')[-1]

    return {
        'type': type_str,
        'value': getattr(cond_obj, 'value', 0),
        'str_val': getattr(cond_obj, 'str_val', ""),
        'stat_key': getattr(cond_obj, 'stat_key', ""),
        'op': getattr(cond_obj, 'op', ""),
        'filter': _convert_filter(getattr(cond_obj, 'filter', None))
    }

def _convert_filter(filt_obj: Any) -> Dict[str, Any]:
    if not filt_obj: return {}

    # Zones enum list
    zones = getattr(filt_obj, 'zones', [])
    zone_strs = [_enum_to_str(z) for z in zones]

    # Types enum list
    types = getattr(filt_obj, 'types', [])
    type_strs = [_enum_to_str(t) for t in types]

    # Civs enum list
    civs = getattr(filt_obj, 'civilizations', [])
    civ_strs = [_enum_to_str(c) for c in civs]

    return {
        'zones': zone_strs,
        'types': type_strs,
        'civilizations': civ_strs,
        'races': list(getattr(filt_obj, 'races', [])),
        'min_cost': getattr(filt_obj, 'min_cost', 0),
        'max_cost': getattr(filt_obj, 'max_cost', 999),
        'min_power': getattr(filt_obj, 'min_power', 0),
        'max_power': getattr(filt_obj, 'max_power', 999999),
        'is_tapped': getattr(filt_obj, 'is_tapped', None),
        'is_blocker': getattr(filt_obj, 'is_blocker', None),
        'is_evolution': getattr(filt_obj, 'is_evolution', None),
        'owner': _enum_to_str(getattr(filt_obj, 'owner', None)),
        'count': getattr(filt_obj, 'count', 0)
    }

def _enum_to_str(val: Any) -> str:
    if val is None: return ""
    if hasattr(val, 'name'):
        return val.name
    return str(val).split('.')[-1]
