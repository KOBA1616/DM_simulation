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
        """Return converted CommandDef/WarningCommand objects for the given legacy action.

        This is a thin adapter that uses the existing conversion pipeline and wraps
        dict outputs into `CommandDef`/`WarningCommand` instances via
        `convert_action_to_objs` to support gradual migration to object-native handling.
        """
        try:
            return convert_action_to_objs(action_data)
        except Exception:
            from dm_toolkit.gui.editor.command_model import WarningCommand
            return [WarningCommand(type="LEGACY_WARNING", warning="Conversion adapter failed", original_action=action_data)]


# Adapter helpers: map legacy-converter dict output -> CommandDef / WarningCommand objects
from dm_toolkit.gui.editor.command_model import CommandDef, WarningCommand
import os
try:
    from dm_toolkit.gui.editor.migration_metrics import record_conversion
except Exception:
    def record_conversion(*args, **kwargs):
        return None


def convert_action_to_objs(action: dict) -> list:
    """Convert a legacy action dict into a list of CommandDef/WarningCommand objects.

    This wraps the existing ActionConverter.convert (which returns dicts) and
    creates object instances to help transition code to the CommandDef model.
    """
    out = []
    try:
        conv = ActionConverter.convert(action)
    except Exception:
        wc = WarningCommand(type="LEGACY_WARNING", warning="Conversion failed", original_action=action)
        return [wc]

    # conv may be a dict representing a command or a legacy-warning object
    if isinstance(conv, dict):
        # detect legacy warning markers
        if conv.get('legacy_warning') or conv.get('type') in (None, 'NONE'):
            wc = WarningCommand(
                type=conv.get('type', 'WARNING'),
                warning=conv.get('str_param', conv.get('warning', 'Legacy conversion produced warning')),
                original_action=conv.get('legacy_original_action') or action,
            )
            out.append(wc)
            try:
                record_conversion(False, action_type=str(conv.get('type')), warning=wc.warning)
            except Exception:
                pass
        else:
            params = {k: v for k, v in conv.items() if k not in ('type', 'uid', 'input_value_key', 'output_value_key', 'if_true', 'if_false', 'options')}
            input_keys = [conv.get('input_value_key')] if conv.get('input_value_key') else []
            output_keys = [conv.get('output_value_key')] if conv.get('output_value_key') else []
            cd = CommandDef(type=conv.get('type', 'UNKNOWN'), params=params, input_keys=input_keys, output_keys=output_keys)
            out.append(cd)
            try:
                record_conversion(True, action_type=str(conv.get('type')))
            except Exception:
                pass
    else:
        # Unknown shape -> warning
        wc = WarningCommand(type='LEGACY_WARNING', warning='Unsupported converter output', original_action=action)
        out.append(wc)

    return out


__all__ = ["convert_action_to_objs"]

# Opt-in: if environment requests native object outputs, make convert return
# dictionary-compatible values while still using object conversion internally.
try:
    if os.environ.get('DM_ACTION_CONVERTER_NATIVE', '0') == '1':
        def _convert_compat(data):
            objs = ActionConverter.convert_to_objs(data)
            # objs is a list of CommandDef/WarningCommand or dicts
            dicts = []
            for o in objs:
                if hasattr(o, 'to_dict'):
                    dicts.append(o.to_dict())
                elif isinstance(o, dict):
                    dicts.append(o)
                else:
                    dicts.append({'type': 'NONE', 'legacy_warning': True, 'legacy_original_action': data})
            # Preserve legacy single-dict return when possible
            if len(dicts) == 1:
                return dicts[0]
            return dicts

        ActionConverter.convert = staticmethod(_convert_compat)
except Exception:
    pass
