# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Union, Optional
import copy
import json
from dm_toolkit.action_to_command import map_action

try:
    from dm_ai_module import ActionDef, CommandDef
except ImportError:
    # Dummy classes for environments without compiled module
    class ActionDef:
        pass
    class CommandDef:
        pass

def to_command_dict(obj: Any) -> Dict[str, Any]:
    """
    Unified entry point to convert any Action-like object to a Command dictionary.

    Args:
        obj: Can be a dict (Legacy Action or Command), ActionDef, or CommandDef.

    Returns:
        A dictionary representing the Command.
    """
    # 1. If it's already a dictionary
    if isinstance(obj, dict):
        return map_action(obj)

    # 2. If it's a CommandDef/ActionDef object with to_json support (e.g. pybind11 structs)
    if hasattr(obj, 'to_json'):
        try:
            # If to_json returns a dict directly (some bindings do this)
            val = obj.to_json()
            if isinstance(val, dict):
                return map_action(val)
            # If to_json returns a string (standard json serialization)
            if isinstance(val, str):
                return map_action(json.loads(val))
        except Exception:
            pass

    # 3. If it's an ActionDef object (Python side class)
    if hasattr(obj, '__dict__'):
        return map_action(obj.__dict__)

    # Fallback: If it's not a dict, not serializable, map_action might fail.
    # We should only call map_action if it looks like a dict or let map_action handle the error.
    # map_action converts to dict internally if possible or returns error command.
    return map_action(obj)

def ensure_executable_command(obj: Any) -> Dict[str, Any]:
    """
    Ensures the given object is a valid Command dictionary ready for execution.
    This is the Unified Execution Path entry point.
    """
    cmd = to_command_dict(obj)

    # Basic Validation (Lightweight)
    if cmd.get('type') == 'NONE' and not cmd.get('legacy_warning'):
        # Maybe it was an empty action or failed conversion
        pass

    return cmd
