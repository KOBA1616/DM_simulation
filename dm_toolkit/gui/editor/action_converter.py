# -*- coding: utf-8 -*-
import uuid
import copy
from dm_toolkit.gui.editor.utils import normalize_action_zone_keys

class ActionConverter:
    @staticmethod
    def convert(action_data):
        """
        Converts a legacy Action dictionary to a Command dictionary.
        """
        # Create a standardized copy to work with
        act_data = normalize_action_zone_keys(action_data.copy())

        act_type = act_data.get('type', 'NONE')
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

        # 1. MOVE_CARD Logic
        if act_type == "MOVE_CARD":
            dest = act_data.get('destination_zone', 'NONE')
            src = act_data.get('source_zone', 'NONE')

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

            ActionConverter._transfer_common_move_fields(act_data, cmd)

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
                 # Use TRANSITION to avoid potential 'Break' logic triggers associated with DESTROY
                 cmd['type'] = "TRANSITION"
                 cmd['from_zone'] = "SHIELD_ZONE"
                 cmd['to_zone'] = "GRAVEYARD"
                 cmd['legacy_warning'] = True
                 cmd['legacy_original_type'] = "SHIELD_BURN"
            else:
                cmd['type'] = act_type # Direct map for DESTROY, DISCARD, RETURN_TO_HAND

            ActionConverter._transfer_common_move_fields(act_data, cmd)

        # 2. DRAW_CARD
        elif act_type == "DRAW_CARD":
            cmd['type'] = "DRAW_CARD"
            cmd['amount'] = act_data.get('value1', 1)
            cmd['target_group'] = act_data.get('scope', 'PLAYER_SELF')
            if 'filter' in act_data:
                cmd['target_filter'] = act_data['filter']

        # 3. TAP / UNTAP
        elif act_type in ["TAP", "UNTAP"]:
            cmd['type'] = act_type
            ActionConverter._transfer_targeting(act_data, cmd)

        # 4. MEASURE (COUNT / GET_STAT)
        elif act_type == "COUNT_CARDS":
            cmd['type'] = "QUERY"
            cmd['str_param'] = "CARDS_MATCHING_FILTER"
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "GET_GAME_STAT":
            cmd['type'] = "QUERY"
            cmd['str_param'] = act_data.get('str_val', '')
            cmd['target_group'] = act_data.get('scope', 'PLAYER_SELF')

        # 5. MODIFIERS
        elif act_type == "APPLY_MODIFIER":
            val = act_data.get('str_val', '')
            if val == "COST":
                cmd['type'] = "MUTATE"
                cmd['mutation_kind'] = "COST"
                cmd['amount'] = act_data.get('value1', 0)
                ActionConverter._transfer_targeting(act_data, cmd)
            else:
                cmd['type'] = "MUTATE"
                cmd['str_param'] = val
                ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "GRANT_KEYWORD":
            cmd['type'] = "ADD_KEYWORD"
            cmd['mutation_kind'] = act_data.get('str_val', '')
            ActionConverter._transfer_targeting(act_data, cmd)
            cmd['amount'] = act_data.get('value1', 1)

        elif act_type == "COST_REDUCTION":
            cmd['type'] = "MUTATE"
            cmd['mutation_kind'] = "COST"
            cmd['amount'] = act_data.get('value1', 0)
            ActionConverter._transfer_targeting(act_data, cmd)

        # 6. SELECT_OPTION (Added)
        elif act_type == "SELECT_OPTION":
            cmd['type'] = "CHOICE"
            # In SELECT_OPTION, value1 is usually irrelevant or default 1, value2 is allow_duplicates
            cmd['amount'] = act_data.get('value1', 1)
            if act_data.get('value2', 0) == 1:
                cmd['flags'] = ["ALLOW_DUPLICATES"]

        # 7. OTHER
        elif act_type == "SEARCH_DECK":
            cmd['type'] = "SEARCH_DECK"
            cmd['amount'] = act_data.get('value1', 1)
            if 'filter' in act_data:
                cmd['target_filter'] = act_data['filter']

        elif act_type == "BREAK_SHIELD":
            cmd['type'] = "BREAK_SHIELD"
            cmd['amount'] = act_data.get('value1', 1)
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "SELECT_TARGET":
            cmd['type'] = "QUERY"
            ActionConverter._transfer_targeting(act_data, cmd)
            if cmd.get('target_group') == 'NONE':
                cmd['target_group'] = 'TARGET_SELECT'

        elif act_type == "MEKRAID":
            # Mekraid is complex (Look top N, Filter, Play).
            cmd['type'] = "NONE"
            cmd['legacy_warning'] = True
            cmd['legacy_original_type'] = "MEKRAID"
            cmd['str_param'] = f"Legacy: {act_type} - Requires manual reconstruction (FLOW)"

        # Fallback for unknown
        else:
            cmd['type'] = "NONE"
            cmd['legacy_warning'] = True
            cmd['legacy_original_type'] = act_type
            cmd['str_param'] = f"Legacy: {act_type}"
            ActionConverter._transfer_targeting(act_data, cmd)

        # Final Check for Critical Missing Fields in standard types
        if not cmd['legacy_warning']:
             if cmd['type'] in ["TRANSITION", "DESTROY", "DISCARD", "RETURN_TO_HAND", "MANA_CHARGE"]:
                   if cmd.get('target_group', 'NONE') == 'NONE':
                       cmd['legacy_warning'] = True
                       cmd['str_param'] = (str(cmd.get('str_param') or '') + " [Missing Target]").strip()

        # Finalize and normalize command (ensure uid, infer basic targets where possible)
        ActionConverter._finalize_command(cmd, act_data)

        return cmd

    @staticmethod
    def _finalize_command(cmd, act):
        """Post-process converted command to improve compatibility and reduce false negatives.

        - Ensure `uid` exists.
        - If a `target_filter` exists but `target_group` is NONE, infer `TARGET_SELECT`.
        - For TRANSITION commands, if `to_zone` exists but `target_group` is NONE, set a reasonable default.
        - If essential fields are missing, leave `legacy_warning` set so callers can handle.
        """
        if 'uid' not in cmd:
            cmd['uid'] = str(uuid.uuid4())

        # Infer target_group from target_filter if possible
        if cmd.get('target_group', 'NONE') == 'NONE' and 'target_filter' in cmd:
            cmd['target_group'] = 'TARGET_SELECT'

        # For TRANSITION, prefer to set target_group when to_zone present
        if cmd.get('type') == 'TRANSITION':
            if 'to_zone' in cmd and cmd.get('target_group', 'NONE') == 'NONE':
                # If the command moves a card to a zone, default to PLAYER_SELF when ambiguous
                cmd['target_group'] = 'PLAYER_SELF'

        # Ensure flags/amount keys exist in normalized form where used
        if 'amount' in act and 'amount' not in cmd:
            try:
                cmd['amount'] = int(act.get('amount') or act.get('value1') or 0)
            except Exception:
                cmd['amount'] = act.get('value1') if 'value1' in act else cmd.get('amount', 0)

        return cmd

    @staticmethod
    def _transfer_targeting(act, cmd):
        scope = act.get('scope', 'NONE')
        # Heuristic Inference: If scope is missing but filter is present, imply TARGET_SELECT
        if scope == 'NONE' and 'filter' in act:
             scope = 'TARGET_SELECT'

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
