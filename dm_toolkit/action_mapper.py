# -*- coding: utf-8 -*-
import uuid
import copy
import warnings
from typing import Any, Dict, Optional, List

# Try to import dm_ai_module to get CommandType enum, otherwise fallback
try:
    import dm_ai_module
    _CommandType = dm_ai_module.CommandType
except ImportError:
    _CommandType = None

try:
    from dm_toolkit.gui.editor.utils import normalize_action_zone_keys
except ImportError:
    # Fallback for runtime environment if gui is not available
    def normalize_action_zone_keys(data):
        if not isinstance(data, dict): return data
        if 'source_zone' not in data and 'from_zone' in data: data['source_zone'] = data['from_zone']
        if 'destination_zone' not in data and 'to_zone' in data: data['destination_zone'] = data['to_zone']
        if 'from_zone' in data: del data['from_zone']
        if 'to_zone' in data: del data['to_zone']
        return data

class ActionToCommandMapper:
    """
    Central logic for converting Action dictionaries to Command dictionaries.
    Used by both the GUI Editor (ActionConverter) and Runtime (commands_new).
    """

    @staticmethod
    def map_action(action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a legacy Action dictionary to a Command dictionary complying with the Schema.
        Handles recursion for nested actions (e.g. options).
        """
        # Defensive copy
        try:
            act_data = action_data.copy() if hasattr(action_data, 'copy') else action_data
        except Exception:
            act_data = action_data

        if not isinstance(act_data, dict):
            # Try to convert object to dict if possible
            if hasattr(act_data, 'to_dict'):
                act_data = act_data.to_dict()
            elif hasattr(act_data, '__dict__'):
                act_data = act_data.__dict__
            else:
                return {
                    "type": "NONE",
                    "uid": str(uuid.uuid4()),
                    "legacy_warning": True,
                    "legacy_original_value": str(act_data),
                    "str_param": "Invalid action shape"
                }

        act_type = str(act_data.get('type', 'NONE'))
        # Handle enum objects
        if hasattr(act_type, 'name'):
            act_type = act_type.name

        cmd = {
            "type": "NONE",
            "uid": str(uuid.uuid4()),
            "legacy_warning": False
        }

        # Common fields transfer
        if 'input_value_key' in act_data:
            cmd['input_value_key'] = act_data['input_value_key']
        if 'output_value_key' in act_data:
            cmd['output_value_key'] = act_data['output_value_key']
        if 'uid' in act_data:
             cmd['uid'] = act_data['uid']

        # Recursive handling of options
        if 'options' in act_data and isinstance(act_data['options'], list):
            cmd['options'] = []
            for opt in act_data['options']:
                if isinstance(opt, list):
                    # List of actions -> List of commands
                    cmd['options'].append([ActionToCommandMapper.map_action(sub) for sub in opt])
                else:
                    # Single action (unlikely structure but safe handling)
                    cmd['options'].append([ActionToCommandMapper.map_action(opt)])

        # Helper to get zone
        def get_zone(d, keys):
            for k in keys:
                if k in d: return d[k]
            return None

        src = get_zone(act_data, ['source_zone', 'from_zone', 'origin_zone'])
        dest = get_zone(act_data, ['destination_zone', 'to_zone', 'dest_zone'])

        # Use strings for CommandTypes to ensure serialization compatibility,
        # relying on the binding enum only if we need integer values (which we generally don't for JSON).
        # We align these strings with C++ NLOHMANN_JSON_SERIALIZE_ENUM

        # 1. MOVE_CARD Logic
        if act_type == "MOVE_CARD":
            if dest == "GRAVEYARD":
                if src == "HAND":
                    cmd['type'] = "DISCARD"
                else:
                    cmd['type'] = "DESTROY"
            elif dest == "MANA_ZONE":
                cmd['type'] = "MANA_CHARGE"
            elif dest == "HAND":
                cmd['type'] = "RETURN_TO_HAND"
            else:
                cmd['type'] = "TRANSITION"
                if dest: cmd['to_zone'] = dest
                if src: cmd['from_zone'] = src

            ActionToCommandMapper._transfer_common_move_fields(act_data, cmd)

        elif act_type in ["DESTROY", "DISCARD", "MANA_CHARGE", "RETURN_TO_HAND", "SEND_TO_MANA", "SEND_TO_DECK_BOTTOM", "ADD_SHIELD", "SHIELD_BURN"]:
            if act_type == "SEND_TO_MANA":
                cmd['type'] = "TRANSITION"
                cmd['to_zone'] = "MANA_ZONE"
            elif act_type == "SEND_TO_DECK_BOTTOM":
                cmd['type'] = "TRANSITION"
                cmd['to_zone'] = "DECK_BOTTOM"
            elif act_type == "ADD_SHIELD":
                cmd['type'] = "TRANSITION"
                cmd['to_zone'] = "SHIELD_ZONE"
            elif act_type == "SHIELD_BURN":
                 cmd['type'] = "SHIELD_BURN" # Assuming this maps to C++
                 cmd['amount'] = act_data.get('value1', 1)
            elif act_type == "DESTROY":
                cmd['type'] = "DESTROY"
            elif act_type == 'RETURN_TO_HAND':
                cmd['type'] = 'RETURN_TO_HAND'
            elif act_type == 'DISCARD':
                cmd['type'] = 'DISCARD'
            elif act_type == 'MANA_CHARGE':
                cmd['type'] = 'MANA_CHARGE'
            else:
                cmd['type'] = act_type

            ActionToCommandMapper._transfer_common_move_fields(act_data, cmd)

        # 2. DRAW_CARD
        elif act_type == "DRAW_CARD":
            cmd['type'] = "DRAW_CARD"
            cmd['from_zone'] = src or 'DECK'
            cmd['to_zone'] = dest or 'HAND'
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            elif 'filter' in act_data and isinstance(act_data['filter'], dict) and 'count' in act_data['filter']:
                cmd['amount'] = act_data['filter']['count']
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # 3. TAP / UNTAP
        elif act_type == "TAP":
            cmd['type'] = "TAP"
            ActionToCommandMapper._transfer_targeting(act_data, cmd)
        elif act_type == "UNTAP":
            cmd['type'] = "UNTAP"
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # 4. MEASURE (COUNT / GET_STAT)
        elif act_type == "COUNT_CARDS" or act_type == "MEASURE_COUNT":
            cmd['type'] = "QUERY"
            cmd['str_param'] = act_data.get('str_val', 'CARDS_MATCHING_FILTER')
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "GET_GAME_STAT":
            cmd['type'] = "QUERY"
            cmd['str_param'] = act_data.get('str_val', '')
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # 5. MODIFIERS
        elif act_type == "APPLY_MODIFIER" or act_type == "COST_REDUCTION":
            val = act_data.get('str_val', '')
            if act_type == "COST_REDUCTION" or val == "COST":
                cmd['type'] = "MUTATE"
                cmd['mutation_kind'] = "COST"
                cmd['amount'] = act_data.get('value1', 0)
            else:
                cmd['type'] = "MUTATE"
                cmd['str_param'] = val
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "GRANT_KEYWORD":
            cmd['type'] = "ADD_KEYWORD"
            cmd['mutation_kind'] = act_data.get('str_val', '')
            cmd['amount'] = act_data.get('value1', 1)
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # 6. MUTATE (Generic legacy)
        elif act_type == "MUTATE":
            sval = str(act_data.get('str_val') or '').upper()
            if sval in ("TAP", "UNTAP"):
                cmd['type'] = sval
            elif sval == "SHIELD_BURN":
                cmd['type'] = "SHIELD_BURN"
                if 'value1' in act_data: cmd['amount'] = act_data['value1']
            elif sval in ("SET_POWER", "POWER_SET"):
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'POWER_SET'
                if 'value1' in act_data: cmd['amount'] = act_data['value1']
            elif 'POWER' in sval or 'POWER_MOD' in sval:
                cmd['type'] = 'POWER_MOD' # Changed from MUTATE/POWER_MOD to direct POWER_MOD macro if supported
                cmd['mutation_kind'] = 'POWER_MOD'
                if 'value1' in act_data: cmd['amount'] = act_data['value1']
                elif 'value2' in act_data: cmd['amount'] = act_data['value2']
            elif 'HEAL' in sval or 'RECOVER' in sval:
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'HEAL'
                if 'value1' in act_data: cmd['amount'] = act_data['value1']
            elif 'REMOVE_KEYWORD' in sval:
                 cmd['type'] = 'MUTATE'
                 cmd['mutation_kind'] = 'REMOVE_KEYWORD'
            else:
                cmd['type'] = "MUTATE"
                cmd['str_param'] = act_data.get('str_val')
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # 7. SELECTION / CHOICE
        elif act_type == "SELECT_OPTION":
            cmd['type'] = "CHOICE" # Engine might not support CHOICE yet, fallback to QUERY?
            # Or map to QUERY with options
            cmd['amount'] = act_data.get('value1', 1)
            if act_data.get('value2', 0) == 1:
                cmd.setdefault('flags', []).append("ALLOW_DUPLICATES")

        elif act_type == "SELECT_NUMBER":
            cmd['type'] = "SELECT_NUMBER"
            if 'value1' in act_data:
                cmd['max'] = int(act_data.get('value1') or 0)
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "SELECT_TARGET":
            cmd['type'] = "QUERY"
            cmd['str_param'] = "SELECT_TARGET" # Implicit
            ActionToCommandMapper._transfer_targeting(act_data, cmd)
            if cmd.get('target_group') == 'NONE':
                cmd['target_group'] = 'TARGET_SELECT'

        # 8. COMPLEX / DECK
        elif act_type == "SEARCH_DECK":
            cmd['type'] = "SEARCH_DECK"
            cmd['amount'] = act_data.get('value1', 1)
            if 'filter' in act_data:
                cmd['target_filter'] = act_data['filter']

        elif act_type == "SHUFFLE_DECK":
            cmd['type'] = "SHUFFLE_DECK" # Engine needs to support this or we assume SEARCH_DECK does it
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "REVEAL_CARDS":
            cmd['type'] = "REVEAL_CARDS"
            cmd['amount'] = act_data.get('value1', 1)
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "LOOK_AND_ADD":
            cmd['type'] = "LOOK_AND_ADD"
            # look count
            if 'value1' in act_data: cmd['look_count'] = int(act_data['value1'])
            elif 'filter' in act_data and 'count' in act_data['filter']: cmd['look_count'] = act_data['filter']['count']
            # add count
            if 'value2' in act_data: cmd['add_count'] = int(act_data['value2'])
            elif 'filter' in act_data and 'select' in act_data['filter']: cmd['add_count'] = act_data['filter']['select']
            # rest zone
            if 'rest_zone' in act_data: cmd['rest_zone'] = act_data['rest_zone']
            elif 'destination_zone' in act_data: cmd['rest_zone'] = act_data['destination_zone']

            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "MEKRAID":
            cmd['type'] = "MEKRAID"
            cmd['look_count'] = int(act_data.get('look_count') or act_data.get('value2') or 3)
            cmd['max_cost'] = act_data.get('value1', 0)
            cmd['select_count'] = 1
            cmd['play_for_free'] = True
            cmd['rest_zone'] = act_data.get('rest_zone') or 'DECK_BOTTOM'
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "REVOLUTION_CHANGE":
            cmd['type'] = "MUTATE"
            cmd['mutation_kind'] = 'REVOLUTION_CHANGE'
            if 'value1' in act_data: cmd['amount'] = act_data['value1']
            if 'str_val' in act_data: cmd['str_param'] = act_data['str_val']
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # 9. PLAY / FLOW
        elif act_type == "PLAY_FROM_ZONE":
            cmd['type'] = "PLAY_FROM_ZONE"
            if src: cmd['from_zone'] = src
            cmd['to_zone'] = dest or 'BATTLE_ZONE'
            if 'value1' in act_data: cmd['max_cost'] = act_data['value1']
            cmd['str_param'] = "PLAY_FROM_ZONE_HINT"
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "FRIEND_BURST":
            cmd['type'] = "FRIEND_BURST"
            cmd['str_val'] = act_data.get('str_val')
            if 'value1' in act_data: cmd['value1'] = act_data['value1']
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "REGISTER_DELAYED_EFFECT":
            cmd['type'] = "REGISTER_DELAYED_EFFECT"
            cmd['str_val'] = act_data.get('str_val')
            if 'value1' in act_data: cmd['value1'] = act_data.get('value1')
            ActionToCommandMapper._transfer_targeting(act_data, cmd)
            if act_data.get('optional', False):
                cmd.setdefault('flags', []).append('OPTIONAL')

        elif act_type == "CAST_SPELL":
            cmd['type'] = "CAST_SPELL"
            ActionToCommandMapper._transfer_targeting(act_data, cmd)
            if 'str_val' in act_data: cmd['str_val'] = act_data['str_val']
            if act_data.get('optional', False):
                cmd.setdefault('flags', []).append('OPTIONAL')

        # 10. Engine Execution / Battle (The "High Priority" items from unmapped)
        elif act_type == "ATTACK_PLAYER":
            cmd['type'] = "ATTACK_PLAYER"
            cmd['instance_id'] = act_data.get('source_instance') or act_data.get('source_instance_id') or act_data.get('attacker_id')
            cmd['target_player'] = act_data.get('target_player')

        elif act_type == "ATTACK_CREATURE":
            cmd['type'] = "ATTACK_CREATURE"
            cmd['instance_id'] = act_data.get('source_instance') or act_data.get('source_instance_id') or act_data.get('attacker_id')
            cmd['target_instance'] = act_data.get('target_instance') or act_data.get('target_instance_id') or act_data.get('target_id')

        elif act_type == "BLOCK":
            cmd['type'] = "FLOW"
            cmd['flow_type'] = "BLOCK"
            cmd['instance_id'] = act_data.get('blocker_id') or act_data.get('source_instance_id')
            cmd['target_instance'] = act_data.get('attacker_id') or act_data.get('target_instance_id')

        elif act_type == "BREAK_SHIELD":
            cmd['type'] = "BREAK_SHIELD"
            cmd['amount'] = act_data.get('value1', 1)
            # engine might provide creature_id, shield_id
            if 'creature_id' in act_data: cmd['instance_id'] = act_data['creature_id']
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "RESOLVE_BATTLE":
            cmd['type'] = "RESOLVE_BATTLE"
            if 'winner_id' in act_data: cmd['winner_instance'] = act_data['winner_id']

        elif act_type == "RESOLVE_EFFECT":
            cmd['type'] = "RESOLVE_EFFECT"
            # Typically wraps an underlying pending effect
            if 'effect_id' in act_data: cmd['effect_id'] = act_data['effect_id']

        elif act_type == "USE_SHIELD_TRIGGER":
            cmd['type'] = "USE_SHIELD_TRIGGER"
            cmd['instance_id'] = act_data.get('card_id') or act_data.get('source_instance_id')

        elif act_type == "RESOLVE_PLAY":
            cmd['type'] = "RESOLVE_PLAY"
            cmd['instance_id'] = act_data.get('card_id') or act_data.get('source_instance_id')

        # 11. Other Buffer Ops
        elif act_type == "LOOK_TO_BUFFER":
            cmd['type'] = 'LOOK_TO_BUFFER'
            cmd['look_count'] = act_data.get('value1', 1)
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "SELECT_FROM_BUFFER":
            cmd['type'] = 'SELECT_FROM_BUFFER'
            cmd['amount'] = act_data.get('value1', 1)
            if act_data.get('value2', 0) == 1: cmd.setdefault('flags', []).append('ALLOW_DUPLICATES')
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "PLAY_FROM_BUFFER":
            cmd['type'] = 'PLAY_FROM_BUFFER'
            cmd['from_zone'] = 'BUFFER'
            cmd['to_zone'] = dest or 'BATTLE_ZONE'
            if 'value1' in act_data: cmd['max_cost'] = act_data['value1']
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "MOVE_BUFFER_TO_ZONE":
            cmd['type'] = 'TRANSITION'
            cmd['from_zone'] = 'BUFFER'
            cmd['to_zone'] = dest or 'HAND'
            if 'value1' in act_data: cmd['amount'] = act_data['value1']
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        elif act_type == "SUMMON_TOKEN":
            cmd['type'] = "SUMMON_TOKEN"
            if 'value1' in act_data: cmd['amount'] = act_data['value1']
            if 'str_val' in act_data: cmd['token_id'] = act_data['str_val']
            ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # Fallback
        else:
            # Special case for "NONE" with str_val (Legacy Keywords)
            if act_type in ("NONE", "") and act_data.get('str_val'):
                 # Try to interpret as keyword grant
                 cmd['type'] = 'ADD_KEYWORD'
                 cmd['mutation_kind'] = str(act_data.get('str_val'))
                 cmd['amount'] = act_data.get('value1', 1)
                 ActionToCommandMapper._transfer_targeting(act_data, cmd)
            else:
                cmd['type'] = "NONE"
                cmd['legacy_warning'] = True
                cmd['legacy_original_type'] = act_type
                cmd['str_param'] = f"Legacy: {act_type}"
                ActionToCommandMapper._transfer_targeting(act_data, cmd)

        # Final Cleanup
        ActionToCommandMapper._finalize_command(cmd, act_data)
        return cmd

    @staticmethod
    def _transfer_targeting(act, cmd):
        scope = act.get('scope', 'NONE')
        if scope == 'NONE' and 'filter' in act:
             scope = 'TARGET_SELECT'
        cmd['target_group'] = scope
        if 'filter' in act:
            cmd['target_filter'] = copy.deepcopy(act['filter'])
        if act.get('optional', False):
            cmd.setdefault('flags', []).append('OPTIONAL')

    @staticmethod
    def _transfer_common_move_fields(act, cmd):
        ActionToCommandMapper._transfer_targeting(act, cmd)
        if 'filter' in act and 'count' in act['filter']:
             cmd['amount'] = act['filter']['count']
        elif 'value1' in act:
             cmd['amount'] = act['value1']

    @staticmethod
    def _finalize_command(cmd, act):
        if 'uid' not in cmd:
            cmd['uid'] = str(uuid.uuid4())

        # Ensure flags/amount keys exist in normalized form
        if 'amount' in act and 'amount' not in cmd:
            try:
                cmd['amount'] = int(act.get('amount') or act.get('value1') or 0)
            except Exception:
                cmd['amount'] = act.get('value1') if 'value1' in act else cmd.get('amount', 0)
