# -*- coding: utf-8 -*-
import uuid
import copy
import warnings
from dm_toolkit.gui.editor.utils import normalize_action_zone_keys

class ActionConverter:
    @staticmethod
    def convert(action_data):
        """
        Converts a legacy Action dictionary to a Command dictionary.
        """
        # Defensive: ensure we have a dict to work with
        try:
            act_data = action_data.copy() if hasattr(action_data, 'copy') else action_data
        except Exception:
            act_data = action_data

        act_data = normalize_action_zone_keys(act_data)

        if not isinstance(act_data, dict):
            # Unexpected shape (primitive/list) — return a legacy-warn command preserving original
            warnings.warn(f"ActionConverter: received non-dict action_data: {type(action_data)}")
            return {
                "type": "NONE",
                "uid": str(uuid.uuid4()),
                "legacy_warning": True,
                "legacy_original_value": action_data,
                "str_param": "Invalid action shape"
            }

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
            # Legacy specific move actions -> prefer TRANSITION where possible
            if act_type == "SEND_TO_MANA":
                # represent as a move into MANA_ZONE
                cmd['type'] = "TRANSITION"
                cmd['to_zone'] = "MANA_ZONE"
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
            elif act_type == "DESTROY":
                # Collapse DESTROY into a TRANSITION to GRAVEYARD
                cmd['type'] = "TRANSITION"
                cmd['to_zone'] = "GRAVEYARD"
            else:
                # DISCARD, RETURN_TO_HAND and other move semantics can be direct TRANSITION
                if act_type == 'RETURN_TO_HAND':
                    cmd['type'] = 'TRANSITION'
                    cmd['to_zone'] = 'HAND'
                elif act_type == 'DISCARD':
                    cmd['type'] = 'TRANSITION'
                    cmd['to_zone'] = 'GRAVEYARD'
                else:
                    # fallback to original command type (e.g., MANA_CHARGE)
                    cmd['type'] = act_type

            ActionConverter._transfer_common_move_fields(act_data, cmd)

        # 2. DRAW_CARD -> unify as TRANSITION (DECK -> HAND)
        elif act_type == "DRAW_CARD":
            # Represent draw as a TRANSITION from DECK to HAND for uniformity
            cmd['type'] = "TRANSITION"
            cmd['from_zone'] = act_data.get('from_zone') or 'DECK'
            cmd['to_zone'] = act_data.get('to_zone') or 'HAND'
            # amount of cards to move
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            elif 'filter' in act_data and isinstance(act_data['filter'], dict) and 'count' in act_data['filter']:
                cmd['amount'] = act_data['filter']['count']
            # preserve filter/targeting
            if 'filter' in act_data:
                cmd['target_filter'] = copy.deepcopy(act_data['filter'])
            cmd['target_group'] = act_data.get('scope', 'PLAYER_SELF')

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

        # Handle legacy MUTATE actions that encode specific operations in `str_val`
        elif act_type == "MUTATE":
            sval = (act_data.get('str_val') or '').upper()
            # Common simple mutations map directly to commands
            if sval in ("TAP", "UNTAP"):
                cmd['type'] = sval
                ActionConverter._transfer_targeting(act_data, cmd)
            elif sval == "SHIELD_BURN":
                # Shield burn can be represented as a dedicated command
                cmd['type'] = "SHIELD_BURN"
                ActionConverter._transfer_targeting(act_data, cmd)
                if 'value1' in act_data:
                    cmd['amount'] = act_data.get('value1')
            elif sval in ("SET_POWER", "POWER_SET"):
                # Explicit set power semantics
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'POWER_SET'
                if 'value1' in act_data:
                    cmd['amount'] = act_data.get('value1')
                ActionConverter._transfer_targeting(act_data, cmd)
            elif sval in ("RESET_POWER", "POWER_RESET"):
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'POWER_RESET'
                ActionConverter._transfer_targeting(act_data, cmd)
            elif 'POWER' in sval or 'POWER_MOD' in sval:
                # Numeric power modification encoded in str_val
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'POWER_MOD'
                if 'value1' in act_data:
                    cmd['amount'] = act_data.get('value1')
                elif 'value2' in act_data:
                    cmd['amount'] = act_data.get('value2')
                ActionConverter._transfer_targeting(act_data, cmd)
            elif 'HEAL' in sval or 'RECOVER' in sval:
                # Legacy heal/recover encoded in mutate strings
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'HEAL'
                if 'value1' in act_data:
                    cmd['amount'] = act_data.get('value1')
                ActionConverter._transfer_targeting(act_data, cmd)
            elif 'REMOVE_KEYWORD' in sval or 'UNGRANT' in sval:
                # Remove granted keyword
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'REMOVE_KEYWORD'
                if 'str_val' in act_data:
                    cmd['str_param'] = act_data.get('str_val')
                ActionConverter._transfer_targeting(act_data, cmd)
            else:
                # Fallback: keep as MUTATE with parameter preserved
                cmd['type'] = "MUTATE"
                if 'str_val' in act_data:
                    cmd['str_param'] = act_data.get('str_val')
                ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "GRANT_KEYWORD":
            cmd['type'] = "ADD_KEYWORD"
            cmd['mutation_kind'] = act_data.get('str_val', '')
            ActionConverter._transfer_targeting(act_data, cmd)
            cmd['amount'] = act_data.get('value1', 1)

        # Support legacy pattern: type == NONE with str_val set to a keyword string
        elif act_type in ("NONE", "") and act_data.get('str_val'):
            sval = str(act_data.get('str_val') or '').strip()
            if sval:
                try:
                    from dm_toolkit.consts import GRANTABLE_KEYWORDS
                    if sval.lower() in [k.lower() for k in GRANTABLE_KEYWORDS]:
                        cmd['type'] = 'ADD_KEYWORD'
                        cmd['mutation_kind'] = sval
                        cmd['amount'] = act_data.get('value1', 1)
                        ActionConverter._transfer_targeting(act_data, cmd)
                        cmd['legacy_warning'] = False
                except Exception:
                    pass

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

        # 6b. SELECT_NUMBER: choose a number (produces an output value)
        elif act_type == "SELECT_NUMBER":
            cmd['type'] = "SELECT_NUMBER"
            # value1 typically indicates the maximum selectable value
            if 'value1' in act_data:
                cmd['max'] = int(act_data.get('value1') or 0)
            # Preserve output key if form attached it
            if 'output_value_key' in act_data:
                cmd['output_value_key'] = act_data.get('output_value_key')
            # Keep targeting if provided (some forms allow scope)
            ActionConverter._transfer_targeting(act_data, cmd)

        # 7. OTHER
        elif act_type == "SEARCH_DECK":
            cmd['type'] = "SEARCH_DECK"
            cmd['amount'] = act_data.get('value1', 1)
            if 'filter' in act_data:
                cmd['target_filter'] = act_data['filter']

        elif act_type == "REVEAL_CARDS":
            cmd['type'] = "REVEAL_CARDS"
            cmd['amount'] = act_data.get('value1', 1)
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "SHUFFLE_DECK":
            cmd['type'] = "SHUFFLE_DECK"
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "BREAK_SHIELD":
            cmd['type'] = "BREAK_SHIELD"
            cmd['amount'] = act_data.get('value1', 1)
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "FRIEND_BURST":
            # Friend-burst: a triggered auxiliary effect linked to a keyword/str_val
            cmd['type'] = "FRIEND_BURST"
            cmd['str_val'] = act_data.get('str_val')
            # value1 may indicate times or count in some legacy forms
            if 'value1' in act_data:
                cmd['value1'] = act_data.get('value1')
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "CAST_SPELL":
            # Cast spell effect: often means 'play this spell-side effect' or 'cast without cost'
            cmd['type'] = "CAST_SPELL"
            # Preserve scope/filter and optional flags
            ActionConverter._transfer_targeting(act_data, cmd)
            if 'str_val' in act_data:
                cmd['str_val'] = act_data.get('str_val')
            if act_data.get('optional', False):
                cmd.setdefault('flags', []).append('OPTIONAL')

        elif act_type == "SELECT_TARGET":
            cmd['type'] = "QUERY"
            ActionConverter._transfer_targeting(act_data, cmd)
            if cmd.get('target_group') == 'NONE':
                cmd['target_group'] = 'TARGET_SELECT'

        elif act_type == "MEKRAID":
            # Mekraid: look top 3, choose a creature with cost <= value1, may summon for free.
            # Represent as a first-class MEKRAID command that preserves the core semantics
            cmd['type'] = "MEKRAID"
            # Default look_count for Mekraid is 3 unless explicitly specified
            look_count = act_data.get('look_count') or act_data.get('value2') or 3
            cmd['look_count'] = int(look_count)
            # value1 commonly encodes the max allowed cost to summon
            if 'value1' in act_data:
                cmd['max_cost'] = act_data.get('value1')
            else:
                cmd['max_cost'] = 0
            # Mekraid typically allows selecting 1 and playing it for free
            cmd['select_count'] = 1
            cmd['play_for_free'] = True
            # preserve any filter and targeting
            if 'filter' in act_data:
                cmd['target_filter'] = copy.deepcopy(act_data['filter'])
            ActionConverter._transfer_targeting(act_data, cmd)
            # default rest placement is deck bottom unless specified
            cmd['rest_zone'] = act_data.get('rest_zone') or 'DECK_BOTTOM'
            # Clear legacy_warning when we can represent the core semantics
            cmd['legacy_warning'] = False

        # 8. REVOLUTION_CHANGE: mark as MUTATE that can be later promoted to keyword+condition
        elif act_type == "REVOLUTION_CHANGE":
                cmd['type'] = "MUTATE"
                # Use mutation_kind to carry semantic meaning; engine-side mapping can interpret
                cmd['mutation_kind'] = "REVOLUTION_CHANGE"
                # Propagate common fields where present
                if 'filter' in act_data:
                    cmd['target_filter'] = copy.deepcopy(act_data['filter'])
                if 'value1' in act_data:
                    # Some legacy REVOLUTION_CHANGE use value1 to indicate magnitude/direction
                    cmd['amount'] = act_data.get('value1')
                if 'str_val' in act_data:
                    cmd['str_param'] = act_data.get('str_val')
                # Consider this a reasonably safe conversion by default

        # 9. PLAY_FROM_ZONE: play a card from a specific zone (no cost)
        elif act_type == "PLAY_FROM_ZONE":
            # Represent as a PLAY hint command: engine/editor can special-case to allow play-from-zone semantics
            cmd['type'] = "PLAY_FROM_ZONE"
            # from_zone/destination may be stored under multiple keys in legacy data
            from_zone = act_data.get('source_zone') or act_data.get('from_zone') or act_data.get('origin_zone')
            to_zone = act_data.get('destination_zone') or act_data.get('to_zone') or 'BATTLE_ZONE'
            if from_zone:
                cmd['from_zone'] = from_zone
            cmd['to_zone'] = to_zone
            # cost hint: value1 often encodes a maximum allowed cost or 0 for free
            if 'value1' in act_data:
                cmd['max_cost'] = act_data.get('value1')
            # preserve scope/filter targeting
            ActionConverter._transfer_targeting(act_data, cmd)
            # mark semantic hint so UI can present it as a play opportunity rather than a raw TRANSITION
            cmd['str_param'] = "PLAY_FROM_ZONE_HINT"

        # 10. MEASURE_COUNT alias: map to QUERY like COUNT_CARDS
        elif act_type == "MEASURE_COUNT":
            cmd['type'] = "QUERY"
            cmd['str_param'] = act_data.get('str_val', 'CARDS_MATCHING_FILTER')
            if 'filter' in act_data:
                cmd['target_filter'] = copy.deepcopy(act_data['filter'])
            cmd['target_group'] = act_data.get('scope', 'PLAYER_SELF')

        # 11. LOOK_AND_ADD: look top N and add matches (map conservatively to TRANSITION to HAND)
        elif act_type == "LOOK_AND_ADD":
            # Map to a LOOK_AND_ADD command capturing reveal/count/rest-zone semantics
            cmd['type'] = "LOOK_AND_ADD"
            # How many to look at (value1 or filter.count)
            look_count = None
            if 'value1' in act_data:
                look_count = act_data.get('value1')
            elif 'filter' in act_data and isinstance(act_data['filter'], dict) and 'count' in act_data['filter']:
                look_count = act_data['filter']['count']
            if look_count is not None:
                cmd['look_count'] = int(look_count)
            # How many to add to hand (value2 or filter.select)
            add_count = None
            if 'value2' in act_data:
                add_count = act_data.get('value2')
            elif 'filter' in act_data and isinstance(act_data['filter'], dict) and 'select' in act_data['filter']:
                add_count = act_data['filter']['select']
            if add_count is not None:
                cmd['add_count'] = int(add_count)
            # remainder destination if specified
            if 'rest_zone' in act_data:
                cmd['rest_zone'] = act_data.get('rest_zone')
            elif 'destination_zone' in act_data:
                cmd['rest_zone'] = act_data.get('destination_zone')
            # preserve filter and targeting
            if 'filter' in act_data:
                cmd['target_filter'] = copy.deepcopy(act_data['filter'])
            ActionConverter._transfer_targeting(act_data, cmd)
            # default to player self
            if cmd.get('target_group', 'NONE') == 'NONE':
                cmd['target_group'] = act_data.get('scope', 'PLAYER_SELF')

        # Buffer-related actions: LOOK_TO_BUFFER, SELECT_FROM_BUFFER, PLAY_FROM_BUFFER, MOVE_BUFFER_TO_ZONE
        elif act_type == "BUFFER_EXTRA":
            # Legacy buffer helper action: perform extra buffer bookkeeping or move several entries
            # Conservatively map to a TRANSITION from BUFFER to a destination zone or to PLAY_FROM_BUFFER hint
            dest = act_data.get('destination_zone') or act_data.get('to_zone') or None
            if dest:
                cmd['type'] = 'TRANSITION'
                cmd['from_zone'] = 'BUFFER'
                cmd['to_zone'] = dest
                if 'value1' in act_data:
                    cmd['amount'] = act_data.get('value1')
            else:
                # No explicit destination: represent as a BUFFER operation hint
                cmd['type'] = 'BUFFER_OP'
                cmd['op'] = act_data.get('str_val') or 'EXTRA'
                if 'value1' in act_data:
                    cmd['amount'] = act_data.get('value1')
            # preserve targeting/filter where useful
            if 'filter' in act_data:
                cmd['target_filter'] = copy.deepcopy(act_data['filter'])
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "LOOK_TO_BUFFER":
            # Look top N into a transient buffer (no immediate add) -- represent as LOOK_TO_BUFFER
            cmd['type'] = 'LOOK_TO_BUFFER'
            cmd['look_count'] = act_data.get('value1', 1)
            if 'filter' in act_data:
                cmd['target_filter'] = copy.deepcopy(act_data['filter'])
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "SELECT_FROM_BUFFER":
            # Select items from buffer into options/choices
            cmd['type'] = 'SELECT_FROM_BUFFER'
            cmd['amount'] = act_data.get('value1', 1)
            # buffer selection may include allow_duplicates in value2
            if act_data.get('value2', 0) == 1:
                cmd.setdefault('flags', []).append('ALLOW_DUPLICATES')
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "PLAY_FROM_BUFFER":
            # Play selected buffer entries (treat as PLAY_FROM_ZONE with buffer origin)
            cmd['type'] = 'PLAY_FROM_BUFFER'
            cmd['from_zone'] = 'BUFFER'
            cmd['to_zone'] = act_data.get('destination_zone', 'BATTLE_ZONE')
            if 'value1' in act_data:
                cmd['max_cost'] = act_data.get('value1')
            ActionConverter._transfer_targeting(act_data, cmd)

        elif act_type == "MOVE_BUFFER_TO_ZONE":
            cmd['type'] = 'TRANSITION'
            # moving items from buffer to a zone
            cmd['from_zone'] = 'BUFFER'
            cmd['to_zone'] = act_data.get('destination_zone', 'HAND')
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            ActionConverter._transfer_common_move_fields(act_data, cmd)

        # Fallback for unknown
        # 13. MOVE_TO_UNDER_CARD: attach a target under a base card
        elif act_type == "MOVE_TO_UNDER_CARD":
            cmd['type'] = "ATTACH"
            # base_target may be stored under several keys in legacy actions
            base = act_data.get('base_target') or act_data.get('to_card') or act_data.get('parent_card')
            if base:
                cmd['base_target'] = base
            # amount/count
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            ActionConverter._transfer_targeting(act_data, cmd)

        # 14. RESET_INSTANCE: reset card instance state (untap, clear attachments etc.)
        elif act_type == "RESET_INSTANCE":
            cmd['type'] = "RESET_INSTANCE"
            ActionConverter._transfer_targeting(act_data, cmd)

        # 15. SUMMON_TOKEN: spawn token(s) with id
        elif act_type == "SUMMON_TOKEN":
            cmd['type'] = "SUMMON_TOKEN"
            # number of tokens
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            # token id/name
            if 'str_val' in act_data:
                cmd['token_id'] = act_data.get('str_val')
            ActionConverter._transfer_targeting(act_data, cmd)

        # SEND_SHIELD_TO_GRAVE: explicitly remove shields to graveyard
        elif act_type == "SEND_SHIELD_TO_GRAVE" or act_type == "SEND_SHIELD_TO_GRAVE":
            cmd['type'] = 'TRANSITION'
            cmd['from_zone'] = 'SHIELD_ZONE'
            cmd['to_zone'] = 'GRAVEYARD'
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            ActionConverter._transfer_targeting(act_data, cmd)

        # PUT_CREATURE: spawn/put a creature onto the battlefield (engine has a handler)
        elif act_type == "PUT_CREATURE":
            # Represent as TRANSITION into BATTLE_ZONE with optional template id
            cmd['type'] = 'TRANSITION'
            cmd['to_zone'] = 'BATTLE_ZONE'
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            if 'str_val' in act_data:
                cmd['str_param'] = act_data.get('str_val')
            # Default to player self when scope not provided to avoid missing-target warnings
            ActionConverter._transfer_targeting(act_data, cmd)
            if cmd.get('target_group', 'NONE') == 'NONE':
                cmd['target_group'] = 'PLAYER_SELF'

        # ADD_MANA: legacy helper to add mana directly
        elif act_type == "ADD_MANA":
            cmd['type'] = 'MANA_CHARGE'
            # amount of mana to add
            cmd['amount'] = act_data.get('value1', 1)
            ActionConverter._transfer_targeting(act_data, cmd)

        # ATTACK actions: map to explicit ATTACK_* command hints preserving source/target
        elif act_type in ("ATTACK_PLAYER", "ATTACK_CREATURE"):
            cmd['type'] = act_type
            # legacy keys may include source_instance / source_instance_id
            if 'source_instance' in act_data:
                cmd['instance_id'] = act_data.get('source_instance')
            elif 'source_instance_id' in act_data:
                cmd['instance_id'] = act_data.get('source_instance_id')
            # preserve explicit target player/instance when present
            if 'target_player' in act_data:
                cmd['target_player'] = act_data.get('target_player')
            if 'target_instance' in act_data:
                cmd['target_instance'] = act_data.get('target_instance')
            ActionConverter._transfer_targeting(act_data, cmd)

        # RESOLVE_BATTLE: engine-level resolution hint
        elif act_type == "RESOLVE_BATTLE":
            cmd['type'] = 'RESOLVE_BATTLE'
            ActionConverter._transfer_targeting(act_data, cmd)

        # FLOW / PHASE change style actions
        elif act_type in ("FLOW", "PHASE_CHANGE"):
            cmd['type'] = 'FLOW'
            # attempt to extract phase/value
            if 'value1' in act_data:
                cmd['value'] = act_data.get('value1')
            elif 'phase' in act_data:
                cmd['value'] = act_data.get('phase')
            # flow_type hint
            cmd['flow_type'] = act_data.get('flow_type') or 'PHASE_CHANGE'
            ActionConverter._transfer_targeting(act_data, cmd)

        # MOVE_TO_DECK_TOP / SEND_TO_DECK_TOP
        elif act_type in ("MOVE_TO_DECK_TOP", "SEND_TO_DECK_TOP"):
            cmd['type'] = 'TRANSITION'
            cmd['to_zone'] = 'DECK_TOP'
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            ActionConverter._transfer_common_move_fields(act_data, cmd)

        # COST_REFERENCE: represent as MUTATE with cost-ref semantics preserved
        elif act_type == "COST_REFERENCE":
            cmd['type'] = 'MUTATE'
            cmd['mutation_kind'] = 'COST_REFERENCE'
            if 'str_val' in act_data:
                cmd['str_param'] = act_data.get('str_val')
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            ActionConverter._transfer_targeting(act_data, cmd)

        # NINJA_STRIKE: map to a play-from-zone hint preserving cost and zone
        elif act_type == "NINJA_STRIKE":
            # Represent as PLAY_FROM_ZONE with ninja hint so editor/engine can special-case
            cmd['type'] = 'PLAY_FROM_ZONE'
            cmd['from_zone'] = act_data.get('zone', 'HAND')
            cmd['to_zone'] = act_data.get('to_zone', 'BATTLE_ZONE')
            if 'cost' in act_data:
                cmd['max_cost'] = act_data.get('cost')
            elif 'value1' in act_data:
                cmd['max_cost'] = act_data.get('value1')
            cmd['str_param'] = 'NINJA_STRIKE'
            # include original for debugging when ambiguous
            cmd['legacy_original'] = act_data
            ActionConverter._transfer_targeting(act_data, cmd)

        # BRANCH: template branching — map to FLOW with branch payload if possible
        elif act_type == "BRANCH":
            cmd['type'] = 'FLOW'
            cmd['flow_type'] = 'BRANCH'
            # copy branch definitions conservatively
            if 'branches' in act_data:
                cmd['branches'] = copy.deepcopy(act_data['branches'])
            elif 'options' in act_data:
                cmd['branches'] = copy.deepcopy(act_data['options'])
            else:
                # fallback: preserve original payload
                cmd['branches'] = act_data.get('payload') or act_data
            ActionConverter._transfer_targeting(act_data, cmd)

        # QUERY: direct pass-through to QUERY command when data contains QUERY action
        elif act_type == "QUERY":
            cmd['type'] = 'QUERY'
            if 'str_val' in act_data:
                cmd['str_param'] = act_data.get('str_val')
            if 'filter' in act_data:
                cmd['target_filter'] = copy.deepcopy(act_data['filter'])
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            ActionConverter._transfer_targeting(act_data, cmd)

        # OPPONENT_DRAW_COUNT: special query for opponent draw counts -> QUERY
        elif act_type == "OPPONENT_DRAW_COUNT":
            cmd['type'] = 'QUERY'
            cmd['str_param'] = 'OPPONENT_DRAW_COUNT'
            if 'value1' in act_data:
                cmd['amount'] = act_data.get('value1')
            ActionConverter._transfer_targeting(act_data, cmd)

        # ADVANCED MUTATE patterns: handle POWER_MOD shorthand and other encoded mutate strings
        elif act_type == 'MUTATE':
            sval = (act_data.get('str_val') or '').upper()
            # If already handled above, keep previous logic; this additional branch handles numeric power mods
            if 'POWER' in sval or 'POWER_MOD' in sval:
                cmd['type'] = 'MUTATE'
                cmd['mutation_kind'] = 'POWER_MOD'
                # amount may be in value1 or value2
                if 'value1' in act_data:
                    cmd['amount'] = act_data.get('value1')
                elif 'value2' in act_data:
                    cmd['amount'] = act_data.get('value2')
                ActionConverter._transfer_targeting(act_data, cmd)
            else:
                # fall back to existing MUTATE handling above (which sets MUTATE or TAP/UNTAP)
                pass

        else:
            cmd['type'] = "NONE"
            cmd['legacy_warning'] = True
            cmd['legacy_original_type'] = act_type
            cmd['str_param'] = f"Legacy: {act_type}"
            ActionConverter._transfer_targeting(act_data, cmd)
        

        # 12. REGISTER_DELAYED_EFFECT: translate to dedicated command to register an effect
        if act_type == "REGISTER_DELAYED_EFFECT":
            # Prefer to represent this as a distinct command that engine recognizes
            cmd['type'] = "REGISTER_DELAYED_EFFECT"
            cmd['str_val'] = act_data.get('str_val')
            # duration is commonly stored in value1
            if 'value1' in act_data:
                cmd['value1'] = act_data.get('value1')
            # transfer targeting if present
            ActionConverter._transfer_targeting(act_data, cmd)
            # optional flags
            if act_data.get('optional', False):
                cmd.setdefault('flags', []).append('OPTIONAL')
            # Clear legacy warning since we map to engine primitive
            cmd['legacy_warning'] = False

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
