"""Shim to expose ActionConverter API used by editor/tests.
Delegates to dm_toolkit.action_to_command.map_action for conversion.
"""
from typing import Any, List
from dm_toolkit.action_to_command import map_action


class ActionConverter:
    @staticmethod
    def convert(action: Any) -> dict:
        """Convert a legacy action to a command dict."""
        cmd = map_action(action)
        # Editor expectations: some MOVE_CARD patterns should canonicalize to DISCARD
        try:
            atype = str(action.get('type') if isinstance(action, dict) else '') .upper()
            src = action.get('source_zone') if isinstance(action, dict) else None
            dst = action.get('destination_zone') or action.get('to_zone') if isinstance(action, dict) else None
            scope = action.get('scope') if isinstance(action, dict) else None
            # If a hand->grave move with PLAYER_SELF scope and a count filter, treat as DISCARD
            if atype == 'MOVE_CARD' and (dst == 'GRAVEYARD') and (src == 'HAND'):
                # convert to DISCARD semantics
                cmd['type'] = 'DISCARD'
                if 'filter' in action and isinstance(action['filter'], dict) and 'count' in action['filter']:
                    cmd['amount'] = action['filter']['count']
        except Exception:
            pass
        return cmd


def convert_action_to_objs(action: Any) -> List[dict]:
    """Return a list of command-like objects for compatibility with editor.
    Most actions map to a single command; wrap result in a list.
    """
    cmd = map_action(action)
    return [cmd]
