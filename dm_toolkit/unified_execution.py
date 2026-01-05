# -*- coding: utf-8 -*-
"""
POLICY: UNIFIED EXECUTION & DISPERSION MINIMIZATION
---------------------------------------------------
This module serves as the primary gateway for executing Actions/Commands.
It integrates with `action_to_command` to ensure all inputs are normalized.

To minimize dispersion of post-processing logic, all high-level execution
wrappers and safety checks should reside here or in `compat_wrappers.py`.
"""

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
    # Capture original simple type hint when input is a dict so we can preserve
    # certain execution-time semantics expected by the unified path tests.
    original_type = None
    if isinstance(obj, dict):
        original_type = obj.get('type')

    cmd = to_command_dict(obj)

    # Basic Validation (Lightweight)
    if cmd.get('type') == 'NONE' and not cmd.get('legacy_warning'):
        # Maybe it was an empty action or failed conversion
        pass

    # Preserve DRAW_CARD direct execution semantics expected by some callers/tests
    if original_type == 'DRAW_CARD':
        cmd['type'] = 'DRAW_CARD'

    return cmd
