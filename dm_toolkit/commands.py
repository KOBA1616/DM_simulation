from typing import Any, Dict, Optional, Protocol, runtime_checkable
from dm_toolkit.action_to_command import map_action

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
    - Otherwise, returns a wrapper that implements `execute` via unified command path
      and `to_dict` via `map_action` from `action_to_command`.
    """
    if action is None:
        return None

    # If it's already command-like, return as-is
    if hasattr(action, "execute") and callable(getattr(action, "execute")):
        return action  # type: ignore

    # Unified wrapper: convert action-like object to command dict and execute via EngineCompat
    class _ActionWrapper(BaseCommand):
        def __init__(self, a: Any):
            self._action = a

        def execute(self, state: Any) -> Optional[Any]:
            try:
                from dm_toolkit.unified_execution import ensure_executable_command
                from dm_toolkit.engine.compat import EngineCompat
                cmd = ensure_executable_command(self._action)
                EngineCompat.ExecuteCommand(state, cmd)
            except Exception:
                return None
            return None

        def invert(self, state: Any) -> Optional[Any]:
            # Best-effort: delegate to underlying object if available
            try:
                inv = getattr(self._action, "invert", None)
                if callable(inv):
                    return inv(state)
            except Exception:
                pass
            return None

        def to_dict(self) -> Dict[str, Any]:
            # Use the unified mapper
            cmd = map_action(self._action)
            return cmd

        def to_string(self) -> str:
            # Check if underlying action has to_string
            if hasattr(self._action, "to_string") and callable(getattr(self._action, "to_string")):
                return self._action.to_string()
            # Fallback to dict description
            d = self.to_dict()
            return str(d)

        def __getattr__(self, name: str) -> Any:
            # Delegate attribute access to underlying action
            return getattr(self._action, name)

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
            # Try to use card_db directly - it could be a native CardDatabase or a dict
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db) or []
        except Exception as e:
            # If it fails, we may have a format mismatch
            # Log for debugging but don't fail - just return empty list
            pass

        cmds = []
        for a in actions:
            cmds.append(wrap_action(a))
        return cmds
    except Exception:
        return []


__all__ = ["ICommand", "BaseCommand", "wrap_action", "generate_legal_commands"]
