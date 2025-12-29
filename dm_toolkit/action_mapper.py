"""Compatibility mapper used by tests that expects slightly different
action->command semantics than the unified mapper.
This file provides `ActionToCommandMapper.map_action`.
"""
from typing import Any
from dm_toolkit.action_to_command import map_action


class ActionToCommandMapper:
    @staticmethod
    def map_action(action: Any) -> dict:
        """Map a legacy action to a command, applying mapper-specific rules.

        This delegates to `map_action` then post-processes to match
        legacy mapper expectations used by unit tests.
        """
        cmd = map_action(action)
        raw_type = str(action.get('type', '')).upper() if isinstance(action, dict) else ''

        # Special-case: MOVE_CARD -> MANA_CHARGE when targeting mana
        to_zone = action.get('to_zone') if isinstance(action, dict) else None
        dest_zone = action.get('destination_zone') if isinstance(action, dict) else None
        if raw_type == 'MOVE_CARD' and (to_zone == 'MANA_ZONE' or dest_zone == 'MANA_ZONE' or cmd.get('to_zone') == 'MANA'):
            cmd['type'] = 'MANA_CHARGE'
            # Ensure to_zone matches legacy expectation
            cmd['to_zone'] = 'MANA_ZONE'

        # Preserve POWER_MOD as explicit type and normalize amount
        if raw_type in ('POWER_MOD', 'MODIFY_POWER'):
            cmd['type'] = 'POWER_MOD'
            if isinstance(action, dict) and 'value1' in action:
                cmd['amount'] = action['value1']

        # Options: some legacy mappers map DRAW_CARD to TRANSITION in options
        if 'options' in cmd and isinstance(cmd['options'], list):
            for group in cmd['options']:
                if isinstance(group, list):
                    for sub in group:
                        if not isinstance(sub, dict):
                            continue
                        orig = str(sub.get('legacy_original_type') or sub.get('type') or '').upper()
                        if orig == 'DRAW_CARD':
                            sub['type'] = 'TRANSITION'

        return cmd
