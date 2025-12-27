from typing import Any, Dict, Optional, Protocol, runtime_checkable
from dm_toolkit.action_mapper import ActionToCommandMapper

@runtime_checkable
class ICommand(Protocol):
    def execute(self, state: Any) -> Optional[Any]:
        ...

    def invert(self, state: Any) -> Optional[Any]:
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...


class BaseCommand:
    """Minimal base command to serve as canonical interface for new commands."""

    def execute(self, state: Any) -> Optional[Any]:
        raise NotImplementedError()

    def invert(self, state: Any) -> Optional[Any]:
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "UNKNOWN", "kind": self.__class__.__name__}


def wrap_action(action: Any) -> Optional[ICommand]:
    """Return an `ICommand`-like object for the provided `action`.

    - If `action` already implements `execute`, return it.
    - Otherwise, returns a wrapper that implements `execute` (via `dm_ai_module` or direct call)
      and `to_dict` (via `ActionToCommandMapper`).
    """
    if action is None:
        return None

    # If it's already command-like, return as-is
    if hasattr(action, "execute") and callable(getattr(action, "execute")):
        return action  # type: ignore

    # Fallback wrapper
    class _ActionWrapper(BaseCommand):
        def __init__(self, a: Any):
            self._action = a

        def execute(self, state: Any) -> Optional[Any]:
            # Try to use action's own execute if available
            if hasattr(self._action, "execute"):
                try:
                    return self._action.execute(state)
                except TypeError:
                    try:
                        return self._action.execute(state, None)
                    except Exception:
                        pass

            # Use shim if available
            try:
                import dm_ai_module
                if hasattr(dm_ai_module, "action_to_command"):
                    cmd = dm_ai_module.action_to_command(self._action)
                    if hasattr(cmd, 'execute'):
                        return cmd.execute(state)
            except Exception:
                pass

            # Use compat shim as last resort (legacy path)
            try:
                from dm_toolkit.compat import ExecuteCommand
                return ExecuteCommand(state, self._action)
            except Exception:
                pass

            return None

        def invert(self, state: Any) -> Optional[Any]:
            try:
                inv = getattr(self._action, "invert", None)
                if callable(inv):
                    return inv(state)
            except Exception:
                pass
            return None

        def to_dict(self) -> Dict[str, Any]:
            # Use the unified mapper
            return ActionToCommandMapper.map_action(self._action)

    return _ActionWrapper(action)


def generate_legal_commands(state: Any, card_db: Dict[int, Any]) -> list:
    """Compatibility helper: generate legal actions and return wrapped commands.

    Calls `dm_ai_module.ActionGenerator.generate_legal_actions` and maps each
    `Action` (or its attached `command`) to an `ICommand` via `wrap_action`.
    """
    try:
        import dm_ai_module

        actions = []
        try:
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db) or []
        except Exception:
            actions = []

        cmds = []
        for a in actions:
            # Prefer explicit attached command if it follows protocol, otherwise wrap action
            cmd_obj = getattr(a, 'command', None)
            if cmd_obj and hasattr(cmd_obj, 'execute'):
                 # Ensure it has to_dict compliant with our schema?
                 # If it's a native C++ command, it might not have Python `to_dict` or might return generic dict.
                 # For now, wrap even the command if it doesn't look like a Python ICommand,
                 # or just wrap the action to be safe and let the wrapper handle execution delegation.
                 # BUT, we want to use the native command logic if present.
                 # Let's wrap the ACTION, because the wrapper prefers action.execute/command.execute but enforces to_dict.
                 cmds.append(wrap_action(a))
            else:
                 cmds.append(wrap_action(a))
        return cmds
    except Exception:
        return []


__all__ = ["ICommand", "BaseCommand", "wrap_action", "generate_legal_commands"]
