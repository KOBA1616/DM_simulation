# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Union, Optional
from collections import UserDict
import copy
import json
from dm_toolkit.action_to_command import map_action
from dm_toolkit.commands import ICommand, BaseCommand

try:
    import dm_ai_module
    from dm_ai_module import ActionDef, CommandDef
except ImportError:
    # Dummy classes for environments without compiled module
    class ActionDef:
        pass
    class CommandDef:
        pass
    dm_ai_module = None

class ExecutableCommand(UserDict, BaseCommand):
    """
    Concrete implementation of ICommand that holds a normalized Command Dictionary
    and knows how to execute it against the Engine (or Compat shim).

    Inherits from UserDict to maintain backward compatibility with consumers
    expecting a raw dictionary (e.g. cmd['type']).
    """
    def __init__(self, command_dict: Dict[str, Any]):
        super().__init__(command_dict)
        self._data = self.data  # Alias for BaseCommand/ICommand compatibility if needed

    def execute(self, state: Any) -> Optional[Any]:
        """
        Execute the command using the best available method on the state/engine.
        """
        # 1. Try direct execution if state supports it (Python Shim or Future C++ binding)
        if hasattr(state, 'execute_command'):
            return state.execute_command(self.data)

        # 2. Try generic C++ command executor if available
        # Note: If dm_ai_module is present and supports generic command execution.
        # Currently the engine mostly uses resolve_action(ActionDef).

        # 3. Fallback to Legacy Action Resolution
        # If the state has resolve_action (e.g. GameInstance, GameState), we try to wrap
        # this command as an ActionDef (if possible) and execute it.
        if hasattr(state, 'resolve_action'):
            # Check if this command comes from a legacy action or is compatible
            # We construct a makeshift ActionDef from the dict
            try:
                # If dm_ai_module is available, use real ActionDef
                if dm_ai_module and hasattr(dm_ai_module, 'ActionDef'):
                     # Mapping from Command Dict back to ActionDef fields might be lossy
                     # but for unified execution of "Legacy-sourced" commands, it works.
                     # We use the raw dict as source.

                     # IMPORTANT: ActionDef constructor might not take **kwargs directly in pybind11.
                     # Usually we need to set attributes.
                     act = dm_ai_module.ActionDef()
                     for k, v in self.data.items():
                         if hasattr(act, k):
                             setattr(act, k, v)

                     return state.resolve_action(act)

                # Mock environment or simple shim
                return state.resolve_action(self.data)
            except Exception:
                # If conversion fails, we can't execute via legacy path.
                pass

        # If we reach here, we cannot execute.
        # For a "Unified Execution" system, raising an error is better than silent failure
        # for explicit execution calls.
        if hasattr(state, 'resolve_action') or hasattr(state, 'execute_command'):
             raise NotImplementedError(f"Could not execute command: {self.data.get('type')}")

        return None

    def to_dict(self) -> Dict[str, Any]:
        return self.data

def to_command_dict(obj: Any) -> Dict[str, Any]:
    """
    Unified entry point to convert any Action-like object to a Command dictionary.
    """
    # 1. If it's already a dictionary
    if isinstance(obj, dict):
        return map_action(obj)

    # 2. If it's a CommandDef/ActionDef object with to_json support
    if hasattr(obj, 'to_json'):
        try:
            val = obj.to_json()
            if isinstance(val, dict):
                return map_action(val)
            if isinstance(val, str):
                return map_action(json.loads(val))
        except Exception:
            pass

    # 3. If it's an ActionDef object
    if hasattr(obj, '__dict__'):
        return map_action(obj.__dict__)

    return map_action(obj)

def ensure_executable_command(obj: Any) -> ICommand:
    """
    Ensures the given object is a valid Executable Command.
    This is the Unified Execution Path entry point.

    Returns an object that:
    1. Implements ICommand (has .execute(state))
    2. Acts like a Dictionary (legacy compat)
    """
    # 1. If it's already an ExecutableCommand (UserDict + ICommand), return as is.
    if isinstance(obj, ExecutableCommand):
        return obj

    # 2. If it's a different ICommand implementation, we return it as is.
    # We generally assume if it has 'execute', it's what we want.
    if isinstance(obj, ICommand) or (hasattr(obj, 'execute') and hasattr(obj, 'to_dict')):
        return obj

    # 3. Convert to normalized Command Dictionary
    cmd_dict = to_command_dict(obj)

    # 4. Wrap in ExecutableCommand
    return ExecutableCommand(cmd_dict)
