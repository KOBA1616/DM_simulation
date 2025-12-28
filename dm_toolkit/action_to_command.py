# -*- coding: utf-8 -*-
import uuid
import copy
from typing import Any, Dict, List, Optional, Union, Protocol, runtime_checkable

# Try to import dm_ai_module to get CommandType enum, otherwise fallback
try:
    import dm_ai_module
    _CommandType = dm_ai_module.CommandType
    _TargetScope = dm_ai_module.TargetScope
except ImportError:
    _CommandType = None
    _TargetScope = None

def normalize_action_zone_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict): return data
    new_data = data.copy()
    if 'source_zone' not in new_data and 'from_zone' in new_data: new_data['source_zone'] = new_data['from_zone']
    if 'destination_zone' not in new_data and 'to_zone' in new_data: new_data['destination_zone'] = new_data['to_zone']
    return new_data

class CommandDefDict(Dict[str, Any]):
    """
    Type hint helper for Command Definition Dictionaries.
    """
    pass

class ActionToCommand:
    """
    Standardized converter from Legacy Action Dictionaries/Objects to new CommandDef Dictionaries.
    """

    @staticmethod
    def map_action(action_data: Union[Dict[str, Any], Any]) -> CommandDefDict:
        """
        Converts a legacy Action to a Command Definition Dictionary.
        The output is compatible with `dm_ai_module.CommandDef` when converted.
        """
        # 1. Standardize Input to Dictionary
        act_data = ActionToCommand._to_dict(action_data)
        if not act_data:
            return ActionToCommand._create_noop("Invalid/Empty Action")

        # 2. Normalize Keys
        act_data = normalize_action_zone_keys(act_data)

        # 3. Determine Type
        act_type = str(act_data.get('type', 'NONE')).upper()
        if hasattr(act_data.get('type'), 'name'):
            act_type = act_data['type'].name.upper()

        # 4. Initialize Command Dict
        cmd: CommandDefDict = {
            "type": "NONE",
            "uid": act_data.get('uid', str(uuid.uuid4()))
        }

        # 5. Map Type and Params
        ActionToCommand._map_type_logic(act_type, act_data, cmd)

        # 6. Transfer Common Properties (Conditions, Metadata)
        ActionToCommand._transfer_common(act_data, cmd)
        ActionToCommand._transfer_options(act_data, cmd)

        return cmd

    @staticmethod
    def _to_dict(action_data: Any) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(action_data, 'to_dict'):
                return action_data.to_dict()
            elif hasattr(action_data, '__dict__'):
                return action_data.__dict__.copy()
            elif isinstance(action_data, dict):
                return copy.deepcopy(action_data)
        except Exception:
            pass
        return None

    @staticmethod
    def _create_noop(reason: str) -> CommandDefDict:
        return {
            "type": "NONE",
            "str_param": reason,
            "legacy_warning": True
        }

    @staticmethod
    def _map_type_logic(act_type: str, act: Dict[str, Any], cmd: Dict[str, Any]):
        src = act.get('source_zone')
        dest = act.get('destination_zone')

        # --- A. TRANSITION / MOVEMENT ---
        if act_type == "MOVE_CARD":
            if dest == "GRAVEYARD":
                if src == "HAND":
                    cmd['type'] = "DISCARD"
                    cmd['from_zone'] = "HAND"
                else:
                    cmd['type'] = "DESTROY"
            elif dest == "MANA_ZONE":
                cmd['type'] = "MANA_CHARGE"
            elif dest == "HAND":
                cmd['type'] = "RETURN_TO_HAND"
            else:
                cmd['type'] = "TRANSITION"

            ActionToCommand._set_zones(cmd, src, dest)
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "DESTROY":
            cmd['type'] = "DESTROY"
            cmd['to_zone'] = "GRAVEYARD"
            ActionToCommand._set_zones(cmd, src, "GRAVEYARD")
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "DISCARD":
            cmd['type'] = "DISCARD"
            cmd['to_zone'] = "GRAVEYARD"
            ActionToCommand._set_zones(cmd, "HAND", "GRAVEYARD")
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "MANA_CHARGE" or act_type == "SEND_TO_MANA":
            cmd['type'] = "MANA_CHARGE"
            ActionToCommand._set_zones(cmd, src or "DECK", "MANA_ZONE")
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "RETURN_TO_HAND":
            cmd['type'] = "RETURN_TO_HAND"
            ActionToCommand._set_zones(cmd, src, "HAND")
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "SEND_TO_DECK_BOTTOM":
            cmd['type'] = "TRANSITION"
            cmd['to_zone'] = "DECK_BOTTOM"
            ActionToCommand._set_zones(cmd, src, None)

        elif act_type == "ADD_SHIELD":
             cmd['type'] = "TRANSITION"
             ActionToCommand._set_zones(cmd, src or "DECK", "SHIELD_ZONE")
             ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "DRAW_CARD":
            cmd['type'] = "DRAW_CARD"
            ActionToCommand._transfer_targeting(act, cmd)

        # --- B. STATE MODIFICATION ---
        elif act_type == "TAP":
            cmd['type'] = "TAP"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "UNTAP":
            cmd['type'] = "UNTAP"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "SHIELD_BURN":
            cmd['type'] = "SHIELD_BURN"
            cmd['amount'] = act.get('value1', 1)
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type in ["APPLY_MODIFIER", "COST_REDUCTION", "GRANT_KEYWORD"]:
            val = act.get('str_val', '')
            if act_type == "GRANT_KEYWORD":
                cmd['type'] = "ADD_KEYWORD"
                cmd['mutation_kind'] = val
            elif act_type == "COST_REDUCTION":
                cmd['type'] = "MUTATE"
                cmd['mutation_kind'] = "COST"
            else:
                cmd['type'] = "MUTATE"
                cmd['mutation_kind'] = val

            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type in ["POWER_MOD", "POWER_SET", "SET_POWER"]:
            cmd['type'] = "POWER_MOD" if act_type == "POWER_MOD" else "MUTATE"
            if act_type != "POWER_MOD":
                cmd['mutation_kind'] = "POWER_SET"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "REVOLUTION_CHANGE":
            cmd['type'] = "MUTATE"
            cmd['mutation_kind'] = 'REVOLUTION_CHANGE'
            if 'value1' in act: cmd['amount'] = act['value1']
            if 'str_val' in act: cmd['str_param'] = act['str_val']
            ActionToCommand._transfer_targeting(act, cmd)

        # --- C. FLOW & QUERIES ---
        elif act_type == "SELECT_TARGET":
            cmd['type'] = "QUERY"
            cmd['str_param'] = "SELECT_TARGET"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "SELECT_NUMBER":
            cmd['type'] = "SELECT_NUMBER"
            cmd['amount'] = act.get('value1')
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "SELECT_OPTION":
            cmd['type'] = "CHOICE"
            cmd['amount'] = act.get('value1', 1)
            if act.get('value2', 0) == 1:
                cmd['str_param'] = "ALLOW_DUPLICATES"
                # Preserve flags for compatibility with potential Python wrappers
                cmd.setdefault('flags', []).append("ALLOW_DUPLICATES")

        elif act_type == "SEARCH_DECK":
            cmd['type'] = "SEARCH_DECK"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "SHUFFLE_DECK":
            cmd['type'] = "SHUFFLE_DECK"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "REVEAL_CARDS":
            cmd['type'] = "REVEAL_CARDS"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "LOOK_AND_ADD":
            cmd['type'] = "LOOK_AND_ADD"
            # Legacy logic transfer
            if 'value1' in act: cmd['look_count'] = int(act['value1'])
            elif 'filter' in act and 'count' in act['filter']: cmd['look_count'] = act['filter']['count']

            if 'value2' in act: cmd['add_count'] = int(act['value2'])
            elif 'filter' in act and 'select' in act['filter']: cmd['add_count'] = act['filter']['select']

            if 'rest_zone' in act: cmd['rest_zone'] = act['rest_zone']
            elif 'destination_zone' in act: cmd['rest_zone'] = act['destination_zone']
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "MEKRAID":
            cmd['type'] = "MEKRAID"
            cmd['look_count'] = int(act.get('look_count') or act.get('value2') or 3)
            cmd['max_cost'] = act.get('value1', 0)
            cmd['rest_zone'] = act.get('rest_zone') or 'DECK_BOTTOM'

            # Legacy mapping restoration
            cmd['select_count'] = 1
            cmd['play_for_free'] = True

            ActionToCommand._transfer_targeting(act, cmd)

        # --- D. PLAY & COMPLEX ---
        elif act_type == "PLAY_FROM_ZONE":
            cmd['type'] = "PLAY_FROM_ZONE"
            ActionToCommand._set_zones(cmd, src, dest or "BATTLE_ZONE")
            if 'value1' in act: cmd['max_cost'] = act['value1']
            cmd['str_param'] = "PLAY_FROM_ZONE_HINT" # Legacy hint restored
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "CAST_SPELL":
            cmd['type'] = "CAST_SPELL"
            if 'str_val' in act: cmd['str_val'] = act['str_val']
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "SUMMON_TOKEN":
            cmd['type'] = "SUMMON_TOKEN"
            if 'str_val' in act: cmd['token_id'] = act['str_val']
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "FRIEND_BURST":
            cmd['type'] = "FRIEND_BURST"
            if 'str_val' in act: cmd['str_val'] = act['str_val']
            if 'value1' in act: cmd['value1'] = act['value1'] # Restored legacy field just in case
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "REGISTER_DELAYED_EFFECT":
            cmd['type'] = "REGISTER_DELAYED_EFFECT"
            ActionToCommand._transfer_targeting(act, cmd)

        # --- E. BUFFER OPS ---
        elif act_type == "LOOK_TO_BUFFER":
            cmd['type'] = "LOOK_TO_BUFFER"
            cmd['look_count'] = act.get('value1', 1)
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "SELECT_FROM_BUFFER":
            cmd['type'] = "SELECT_FROM_BUFFER"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "PLAY_FROM_BUFFER":
            cmd['type'] = "PLAY_FROM_BUFFER"
            ActionToCommand._set_zones(cmd, "BUFFER", dest or "BATTLE_ZONE")
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "MOVE_BUFFER_TO_ZONE":
            cmd['type'] = "MOVE_BUFFER_TO_ZONE" # Or TRANSITION from BUFFER
            # CommandType has MOVE_BUFFER_TO_ZONE
            ActionToCommand._set_zones(cmd, "BUFFER", dest or "HAND")
            ActionToCommand._transfer_targeting(act, cmd)

        # --- F. ENGINE / BATTLE ---
        elif act_type == "ATTACK_PLAYER":
            cmd['type'] = "ATTACK_PLAYER"
            cmd['instance_id'] = act.get('source_instance') or act.get('source_instance_id') or act.get('attacker_id')
            cmd['target_player'] = act.get('target_player')

        elif act_type == "ATTACK_CREATURE":
            cmd['type'] = "ATTACK_CREATURE"
            cmd['instance_id'] = act.get('source_instance') or act.get('source_instance_id') or act.get('attacker_id')
            cmd['target_instance'] = act.get('target_instance') or act.get('target_instance_id') or act.get('target_id')

        elif act_type == "BLOCK":
            cmd['type'] = "BLOCK"
            cmd['instance_id'] = act.get('blocker_id') or act.get('source_instance_id')
            cmd['target_instance'] = act.get('attacker_id') or act.get('target_instance_id')

        elif act_type == "BREAK_SHIELD":
            cmd['type'] = "BREAK_SHIELD"
            ActionToCommand._transfer_targeting(act, cmd)

        elif act_type == "RESOLVE_BATTLE":
            cmd['type'] = "RESOLVE_BATTLE"
            if 'winner_id' in act: cmd['winner_instance'] = act['winner_id']

        elif act_type == "RESOLVE_EFFECT":
            cmd['type'] = "RESOLVE_EFFECT"
            if 'effect_id' in act: cmd['effect_id'] = act['effect_id']

        elif act_type == "RESOLVE_PLAY":
            cmd['type'] = "RESOLVE_PLAY"
            cmd['instance_id'] = act.get('card_id') or act.get('source_instance_id')

        # --- G. FALLBACK ---
        else:
            # Legacy keyword handling
            if act_type in ("NONE", "") and act.get('str_val'):
                 cmd['type'] = 'ADD_KEYWORD'
                 cmd['mutation_kind'] = str(act.get('str_val'))
                 ActionToCommand._transfer_targeting(act, cmd)
            else:
                cmd['type'] = "NONE"
                cmd['str_param'] = f"Unknown: {act_type}"
                cmd['legacy_warning'] = True
                ActionToCommand._transfer_targeting(act, cmd)

    @staticmethod
    def _set_zones(cmd, src, dest):
        if src: cmd['from_zone'] = src
        if dest: cmd['to_zone'] = dest

    @staticmethod
    def _transfer_targeting(act, cmd):
        # 1. Amount
        if 'amount' in act:
            cmd['amount'] = int(act['amount'])
        elif 'value1' in act:
            cmd['amount'] = int(act['value1'])
        elif 'filter' in act and isinstance(act['filter'], dict) and 'count' in act['filter']:
            cmd['amount'] = int(act['filter']['count'])

        # 2. Filter / Target Group
        scope = act.get('scope', 'NONE')
        if scope == 'NONE' and 'filter' in act:
             scope = 'TARGET_SELECT'

        cmd['target_group'] = scope

        if 'filter' in act:
            cmd['target_filter'] = copy.deepcopy(act['filter'])

        # 3. Optional
        if act.get('optional', False):
            cmd['optional'] = True

        # 4. Keys
        if 'input_value_key' in act: cmd['input_value_key'] = act['input_value_key']
        if 'output_value_key' in act: cmd['output_value_key'] = act['output_value_key']

    @staticmethod
    def _transfer_common(act, cmd):
        pass # Metadata if needed

    @staticmethod
    def _transfer_options(act, cmd):
        # Recursively handle options
        # ActionDef: options is List[List[ActionDef]]
        # CommandDef: if_true, if_false (List[CommandDef])
        # We map options[0] -> if_true, options[1] -> if_false if they exist?
        # Or does CommandDef have options? card_json_types.hpp says CommandDef has if_true/if_false, NOT options.
        # But ActionDef has options.
        # CHOICE action usually implies picking an option index which then executes that option branch.
        # In CommandDef, CHOICE might rely on `if_true` being Option 0 and `if_false` being Option 1?
        # Or maybe CommandDef structure needs extension.
        # For now, we map the first two options to if_true/if_false to preserve as much as possible.
        if 'options' in act and isinstance(act['options'], list):
            if len(act['options']) > 0 and isinstance(act['options'][0], list):
                 # Map first option branch to if_true
                 cmd['if_true'] = [ActionToCommand.map_action(sub) for sub in act['options'][0]]

            if len(act['options']) > 1 and isinstance(act['options'][1], list):
                 # Map second option branch to if_false
                 cmd['if_false'] = [ActionToCommand.map_action(sub) for sub in act['options'][1]]
