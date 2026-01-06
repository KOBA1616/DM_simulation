# -*- coding: utf-8 -*-
from typing import Any, Dict
from dm_toolkit.action_to_command import map_action as _map_action

"""
DEPRECATED: Use dm_toolkit.unified_execution instead.

This module provides legacy mapping logic that is being phased out in favor of
the Unified Execution Pipeline.
"""

import warnings

class ActionToCommandMapper:
    """
    Central logic for converting Action dictionaries to Command dictionaries.
    Wrapper around the pure function implementation in action_to_command.py
    """

    @staticmethod
    def map_action(action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a legacy Action dictionary to a Command dictionary.
        """
        warnings.warn(
            "ActionToCommandMapper is deprecated. Use dm_toolkit.action_to_command.map_action directly.",
            DeprecationWarning,
            stacklevel=2
        )
        # Use the canonical mapper first
        cmd = _map_action(action_data)

        # Post-process to preserve legacy expectations required by some callers/tests.
        try:
            atype = str(action_data.get('type') if isinstance(action_data, dict) else '') .upper()
        except Exception:
            atype = ''

        # 1) Legacy MOVE_CARD -> MANA_CHARGE expectation when destination was MANA_ZONE
        if atype == 'MOVE_CARD':
            # use original keys if present to preserve legacy naming
            dest = None
            src = None
            if isinstance(action_data, dict):
                dest = action_data.get('to_zone') or action_data.get('destination_zone')
                src = action_data.get('from_zone') or action_data.get('source_zone')
            # If original destination explicitly used MANA_ZONE, map to MANA_CHARGE and keep original zone naming
            if dest == 'MANA_ZONE' or dest == 'MANA':
                cmd['type'] = 'MANA_CHARGE'
                # Preserve legacy zone strings when available
                if dest is not None:
                    cmd['to_zone'] = dest
                if src is not None:
                    cmd['from_zone'] = src

        # 2) Numeric POWER_MOD should preserve type name in legacy mapper
        if atype == 'POWER_MOD' or (isinstance(action_data, dict) and str(action_data.get('str_val','')).upper() == 'POWER_MOD'):
            cmd['type'] = 'POWER_MOD'
            if isinstance(action_data, dict) and 'value1' in action_data:
                cmd['amount'] = action_data.get('value1')

        # 4) PLAY_FROM_ZONE: preserve original destination zone naming if provided
        if atype == 'PLAY_FROM_ZONE' and isinstance(action_data, dict):
            dest = action_data.get('destination_zone') or action_data.get('to_zone')
            if dest:
                cmd['to_zone'] = dest

        # 5) MOVE_TO_UNDER_CARD should be represented as ATTACH in legacy converter
        if atype == 'MOVE_TO_UNDER_CARD':
            cmd['type'] = 'ATTACH'
            # preserve original base_target field if present
            if isinstance(action_data, dict) and 'base_target' in action_data:
                cmd['base_target'] = action_data.get('base_target')

        # 6) RESET_INSTANCE: some callers expect the literal RESET_INSTANCE command
        if atype == 'RESET_INSTANCE':
            cmd['type'] = 'RESET_INSTANCE'

        # 3) Options recursion: some tests expect DRAW_CARD inside options to be treated as TRANSITION
        if 'options' in cmd and isinstance(cmd['options'], list):
            for opt_group in cmd['options']:
                if isinstance(opt_group, list):
                    for sub in opt_group:
                        if sub.get('type') == 'DRAW_CARD':
                            sub['type'] = 'TRANSITION'
                            # ensure amount is set
                            if 'amount' not in sub and 'value1' in sub:
                                sub['amount'] = sub['value1']

        return cmd
