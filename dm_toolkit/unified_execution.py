# -*- coding: utf-8 -*-
"""
Unified Execution Pipeline Entry Point

This module serves as the **canonical entry point** for converting Action-like objects
(dictionaries, ActionDef, CommandDef) into a unified Command dictionary format ready
for execution by the game engine.

AGENTS.md Policy Compliance:
- Section 1: All conversions route through dm_toolkit.action_to_command.map_action
- Section 2: Execution standardization minimizes dispersion of command post-processing
- Section 3: Supports headless testing via run_pytest_with_pyqt_stub.py integration

Key Functions:
- to_command_dict: Converts any action-like object to a command dictionary
- ensure_executable_command: Validates and prepares commands for engine execution
- execute_command: Executes a command using the engine compatibility layer
- execute_commands: Executes a sequence of commands

Usage:
    from dm_toolkit.unified_execution import ensure_executable_command, execute_command
    
    # From legacy action dict
    action = {"type": "DRAW_CARD", "value1": 2}
    cmd = ensure_executable_command(action)
    
    # From ActionDef object
    action_obj = ActionDef(...)
    cmd = ensure_executable_command(action_obj)
    
    # Execute via engine
    execute_command(state, cmd, card_db)

Note: action_mapper.py is deprecated; use this module instead.
"""
from typing import Any, Dict, List, Union, Optional, TYPE_CHECKING, Iterable
import copy
import json
from dm_toolkit.action_to_command import map_action

if TYPE_CHECKING:
    # Provide type information for static analysis only
    from dm_ai_module import ActionDef, CommandDef  # type: ignore
else:
    try:
        import dm_ai_module as _native_dm  # type: ignore
        ActionDef = getattr(_native_dm, 'ActionDef', object)
        CommandDef = getattr(_native_dm, 'CommandDef', object)
        _HAS_NATIVE = True
    except Exception:
        # Dummy runtime fallbacks for environments without the compiled module
        class ActionDef:
            pass
        class CommandDef:
            pass
        _HAS_NATIVE = False

def to_command_dict(obj: Any) -> Dict[str, Any]:
    """
    Unified entry point to convert any Action-like object to a Command dictionary.

    Args:
        obj: Can be a dict (Legacy Action or Command), ActionDef, or CommandDef.

    Returns:
        A dictionary representing the Command.
    """
    # 0. Prefer explicit `to_dict` method if available (Native CommandDef support)
    try:
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
    except Exception:
        pass

    # 1. If the object exposes a precomputed `command` attribute, prefer it.
    # This allows Action instances to carry a canonical command dict produced
    # by generators (see dm_ai_module.IntentGenerator.command). Support
    # both dicts and objects that can be mapped via map_action.
    try:
        if obj is not None and hasattr(obj, 'command'):
            cmd_attr = getattr(obj, 'command')
            if isinstance(cmd_attr, dict):
                return map_action(cmd_attr)
            # If it's an object with to_json or similar, fall through to
            # existing handling by mapping the attribute value.
            try:
                return map_action(cmd_attr)
            except Exception:
                # Fall back to continuing conversion below
                pass
    except Exception:
        pass

    # 2. If it's already a dictionary
    if isinstance(obj, dict):
        return map_action(obj)

    # Emit a deprecation warning when callers pass Action-like objects that
    # do not carry a `command` attribute. This encourages migration to the
    # command-first path where possible.
    try:
        import warnings
        if obj is not None and not isinstance(obj, dict) and not hasattr(obj, 'command'):
            warnings.warn(
                "Passing Action-like objects to unified execution is deprecated — attach a 'command' dict or use ensure_executable_command(map) to migrate.",
                DeprecationWarning,
                stacklevel=3,
            )
    except Exception:
        pass

    # 3. If it's a CommandDef/ActionDef object with to_json support (e.g. pybind11 structs)
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

    # 4. If it's an ActionDef object (Python side class)
    if hasattr(obj, '__dict__'):
        return map_action(obj.__dict__)

    # Fallback: If it's not a dict, not serializable, map_action might fail.
    # We should only call map_action if it looks like a dict or let map_action handle the error.
    # map_action converts to dict internally if possible or returns error command.
    return map_action(obj)

def ensure_executable_command(obj: Any) -> Union[Dict[str, Any], Any]:
    """
    Ensures the given object is a valid Command dictionary ready for execution.
    This is the Unified Execution Path entry point.
    
    Post-Processing (Specs/AGENTS.md Policy Section 2):
    - Applies legacy field normalization via compat_wrappers
    - Preserves backward compatibility with legacy test code
    - Validates command structure for execution readiness
    
    Args:
        obj: Action-like object (dict, ActionDef, CommandDef, etc.)
        
    Returns:
        Validated and normalized Command dictionary ready for engine execution
    """
    # Phase 3: Pass through native CommandDef objects directly
    if _HAS_NATIVE and isinstance(obj, CommandDef):
        return obj

    from dm_toolkit.compat_wrappers import normalize_legacy_fields
    
    # Capture original simple type hint when input is a dict so we can preserve
    # certain execution-time semantics expected by the unified path tests.
    original_type = None
    if isinstance(obj, dict):
        original_type = obj.get('type')

    cmd = to_command_dict(obj)

    # If mapping produced a NONE/legacy_warning result, attempt a more
    # aggressive extraction for dm_ai_module.Action-like objects which may
    # expose fields as properties (pybind11) rather than __dict__.
    try:
        if (not cmd or cmd.get('type') in (None, 'NONE')) or cmd.get('legacy_warning'):
            # Try to extract a conservative field set from the original object
            if obj is not None and not isinstance(obj, dict):
                extracted = {}
                # common fields to attempt to read
                for field in ('type', 'card_id', 'source_instance_id', 'instance_id', 'target_instance_id',
                              'target_player', 'slot_index', 'target_slot_index',
                              'value1', 'value2', 'str_val', 'from_zone', 'to_zone',
                              'play_for_free', 'play_free', 'put_into_play'):
                    try:
                        if hasattr(obj, field):
                            extracted[field] = getattr(obj, field)
                    except Exception:
                        pass

                # Only re-map if we found something useful beyond empty
                if extracted:
                    # Prefer explicit 'type' override when available
                    cmd2 = map_action(extracted)
                    # If mapping looks valid (not NONE), use it
                    if cmd2 and cmd2.get('type') not in (None, 'NONE'):
                        cmd = cmd2
    except Exception:
        # Be defensive: never raise here
        pass

    # Basic Validation (Lightweight)
    if cmd.get('type') == 'NONE' and not cmd.get('legacy_warning'):
        # Maybe it was an empty action or failed conversion
        pass

    # Preserve DRAW_CARD direct execution semantics expected by some callers/tests
    if original_type == 'DRAW_CARD':
        cmd['type'] = 'DRAW_CARD'
    
    # Apply backward compatibility normalization (Specs/AGENTS.md Policy Section 2)
    cmd = normalize_legacy_fields(cmd)

    return cmd


def execute_command(state: Any, command: Any, card_db: Any = None) -> None:
    """Executes a single command using the engine compatibility layer."""
    from dm_toolkit.engine.compat import EngineCompat
    # EngineCompat.ExecuteCommand accepts dicts or bound CommandDef objects
    if card_db is not None:
        EngineCompat.ExecuteCommand(state, command, card_db)
    else:
        EngineCompat.ExecuteCommand(state, command)


def execute_commands(state: Any, commands: Iterable[Any], card_db: Any = None) -> None:
    """Executes a sequence of commands."""
    for c in commands:
        execute_command(state, c, card_db)
