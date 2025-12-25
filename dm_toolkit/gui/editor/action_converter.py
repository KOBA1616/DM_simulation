# -*- coding: utf-8 -*-
import uuid
import copy

class ActionConverter:
    @staticmethod
    def convert(action_data):
        """
        Converts a legacy Action dictionary to a Command dictionary.
        """
        act_type = action_data.get('type', 'NONE')
        cmd = {
            "type": "NONE",
            "uid": str(uuid.uuid4())
        }

        # Common fields transfer
        if 'input_value_key' in action_data:
            cmd['input_value_key'] = action_data['input_value_key']
        if 'output_value_key' in action_data:
            cmd['output_value_key'] = action_data['output_value_key']

        # 1. MOVE_CARD Logic
        if act_type == "MOVE_CARD":
            dest = action_data.get('destination_zone', 'NONE')
            src = action_data.get('source_zone', 'NONE')

            # Handle possible legacy keys from bad config templates
            if dest == 'NONE' and 'to_zone' in action_data:
                 dest = action_data['to_zone']
            if src == 'NONE' and 'from_zone' in action_data:
                 src = action_data['from_zone']

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
                cmd['to_zone'] = dest
                if src != 'NONE':
                    cmd['from_zone'] = src

            ActionConverter._transfer_common_move_fields(action_data, cmd)

        elif act_type in ["DESTROY", "DISCARD", "MANA_CHARGE", "RETURN_TO_HAND", "SEND_TO_MANA", "SEND_TO_DECK_BOTTOM", "ADD_SHIELD", "SHIELD_BURN"]:
            # Legacy specific move actions
            if act_type == "SEND_TO_MANA":
                cmd['type'] = "MANA_CHARGE"
            elif act_type == "SEND_TO_DECK_BOTTOM":
                cmd['type'] = "TRANSITION"
                cmd['to_zone'] = "DECK_BOTTOM"
            elif act_type == "ADD_SHIELD":
                cmd['type'] = "TRANSITION"
                cmd['to_zone'] = "SHIELD_ZONE"
            elif act_type == "SHIELD_BURN":
                 cmd['type'] = "DESTROY" # Close enough, or TRANSITION
                 cmd['from_zone'] = "SHIELD_ZONE"
                 cmd['to_zone'] = "GRAVEYARD"
            else:
                cmd['type'] = act_type # Direct map for DESTROY, DISCARD, RETURN_TO_HAND

            ActionConverter._transfer_common_move_fields(action_data, cmd)

        # 2. DRAW_CARD
        elif act_type == "DRAW_CARD":
            cmd['type'] = "DRAW_CARD"
            cmd['amount'] = action_data.get('value1', 1)
            cmd['target_group'] = action_data.get('scope', 'PLAYER_SELF')
            if 'filter' in action_data:
                cmd['target_filter'] = action_data['filter']

        # 3. TAP / UNTAP
        elif act_type in ["TAP", "UNTAP"]:
            cmd['type'] = act_type
            ActionConverter._transfer_targeting(action_data, cmd)

        # 4. MEASURE (COUNT / GET_STAT)
        elif act_type == "COUNT_CARDS":
            cmd['type'] = "QUERY"
            cmd['str_param'] = "CARDS_MATCHING_FILTER"
            ActionConverter._transfer_targeting(action_data, cmd)

        elif act_type == "GET_GAME_STAT":
            cmd['type'] = "QUERY"
            cmd['str_param'] = action_data.get('str_val', '')
            cmd['target_group'] = action_data.get('scope', 'PLAYER_SELF')

        # 5. MODIFIERS
        elif act_type == "APPLY_MODIFIER":
            val = action_data.get('str_val', '')
            if val == "COST":
                cmd['type'] = "MUTATE"
                cmd['mutation_kind'] = "COST"
                cmd['amount'] = action_data.get('value1', 0)
                ActionConverter._transfer_targeting(action_data, cmd)
            else:
                cmd['type'] = "MUTATE"
                cmd['str_param'] = val
                ActionConverter._transfer_targeting(action_data, cmd)

        elif act_type == "GRANT_KEYWORD":
            cmd['type'] = "ADD_KEYWORD"
            cmd['mutation_kind'] = action_data.get('str_val', '')
            ActionConverter._transfer_targeting(action_data, cmd)
            cmd['amount'] = action_data.get('value1', 1)

        elif act_type == "COST_REDUCTION":
            cmd['type'] = "MUTATE"
            cmd['mutation_kind'] = "COST"
            cmd['amount'] = action_data.get('value1', 0)
            ActionConverter._transfer_targeting(action_data, cmd)

        # 6. SELECT_OPTION (Added)
        elif act_type == "SELECT_OPTION":
            cmd['type'] = "CHOICE"
            # In SELECT_OPTION, value1 is usually irrelevant or default 1, value2 is allow_duplicates
            cmd['amount'] = action_data.get('value1', 1)
            if action_data.get('value2', 0) == 1:
                cmd['flags'] = ["ALLOW_DUPLICATES"]

        # 7. OTHER
        elif act_type == "SEARCH_DECK":
            cmd['type'] = "SEARCH_DECK"
            cmd['amount'] = action_data.get('value1', 1)
            if 'filter' in action_data:
                cmd['target_filter'] = action_data['filter']

        elif act_type == "BREAK_SHIELD":
            cmd['type'] = "BREAK_SHIELD"
            cmd['amount'] = action_data.get('value1', 1)
            ActionConverter._transfer_targeting(action_data, cmd)

        elif act_type == "SELECT_TARGET":
            cmd['type'] = "QUERY"
            ActionConverter._transfer_targeting(action_data, cmd)
            if cmd.get('target_group') == 'NONE':
                cmd['target_group'] = 'TARGET_SELECT'

        elif act_type == "MEKRAID":
            # Mekraid is complex (Look top N, Filter, Play).
            cmd['type'] = "NONE"
            cmd['str_param'] = f"Legacy: {act_type}"

        # Fallback for unknown
        else:
            cmd['type'] = "NONE"
            cmd['str_param'] = f"Legacy: {act_type}"
            ActionConverter._transfer_targeting(action_data, cmd)

        return cmd

    @staticmethod
    def _transfer_targeting(act, cmd):
        scope = act.get('scope', 'NONE')
        cmd['target_group'] = scope
        if 'filter' in act:
            cmd['target_filter'] = copy.deepcopy(act['filter'])

    @staticmethod
    def _transfer_common_move_fields(act, cmd):
        ActionConverter._transfer_targeting(act, cmd)

        if 'filter' in act and 'count' in act['filter']:
             cmd['amount'] = act['filter']['count']
        elif 'value1' in act:
             cmd['amount'] = act['value1']

        cmd['optional'] = act.get('optional', False)
