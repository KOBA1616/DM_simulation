# -*- coding: utf-8 -*-
import uuid
import warnings
from dm_toolkit.action_mapper import ActionToCommandMapper
from dm_toolkit.gui.editor.utils import normalize_action_zone_keys

class ActionConverter:
    @staticmethod
    def convert(action_data):
        """
        Converts a legacy Action dictionary to a Command dictionary.
        Delegates to the shared ActionToCommandMapper to ensure consistency.
        """
        return ActionToCommandMapper.map_action(action_data)

    @staticmethod
    def convert_to_objs(action_data):
        """Return converted CommandDef/WarningCommand objects for the given legacy action."""
        try:
            return convert_action_to_objs(action_data)
        except Exception:
            from dm_toolkit.gui.editor.command_model import WarningCommand
            return [WarningCommand(type="LEGACY_WARNING", warning="Conversion adapter failed", original_action=action_data)]


from dm_toolkit.gui.editor.command_model import CommandDef, WarningCommand
import os
try:
    from dm_toolkit.gui.editor.migration_metrics import record_conversion
except Exception:
    def record_conversion(*args, **kwargs):
        return None


def convert_action_to_objs(action: dict) -> list:
    """Convert a legacy action dict into a list of CommandDef/WarningCommand objects."""
    out = []
    try:
        conv = ActionConverter.convert(action)
    except Exception:
        wc = WarningCommand(type="LEGACY_WARNING", warning="Conversion failed", original_action=action)
        return [wc]

    # conv is a dict (flat command) or a dict indicating warning
    if isinstance(conv, dict):
        if conv.get('legacy_warning') or conv.get('type') in (None, 'NONE'):
            wc = WarningCommand.from_dict(conv)
            # Ensure original action is attached if not present
            if not wc.original_action:
                wc.original_action = action
            out.append(wc)
            try:
                record_conversion(False, action_type=str(conv.get('type')), warning=wc.warning)
            except Exception:
                pass
        else:
            # Use CommandDef.from_dict to parse the flat dict directly
            cd = CommandDef.from_dict(conv)
            out.append(cd)
            try:
                record_conversion(True, action_type=str(conv.get('type')))
            except Exception:
                pass
    else:
        wc = WarningCommand(type='LEGACY_WARNING', warning='Unsupported converter output', original_action=action)
        out.append(wc)

    return out


__all__ = ["convert_action_to_objs"]

try:
    if os.environ.get('DM_ACTION_CONVERTER_NATIVE', '0') == '1':
        def _convert_compat(data):
            # Use the standalone function instead of the class method to be safe,
            # though the class method is restored above.
            objs = convert_action_to_objs(data)
            dicts = []
            for o in objs:
                if hasattr(o, 'to_dict'):
                    dicts.append(o.to_dict())
                elif isinstance(o, dict):
                    dicts.append(o)
                else:
                    dicts.append({'type': 'NONE', 'legacy_warning': True, 'legacy_original_action': data})
            if len(dicts) == 1:
                return dicts[0]
            return dicts

        ActionConverter.convert = staticmethod(_convert_compat)
except Exception:
    pass
