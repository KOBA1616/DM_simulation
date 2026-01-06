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
    - Otherwise, returns a wrapper that implements `execute` (via `dm_ai_module` or direct call)
      and `to_dict` (via `map_action` from `action_to_command`).
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
            cmd = map_action(self._action)
            # Legacy wrapper: when wrapping an Action object representing a DRAW_CARD,
            # some callers expect it to be represented as a TRANSITION for compatibility.
            try:
                # Determine original type from the action object if available
                orig_type = None
                if hasattr(self._action, 'type'):
                    orig_type = str(getattr(self._action, 'type')).upper()
                elif isinstance(self._action, dict):
                    orig_type = str(self._action.get('type', '')).upper()

                if orig_type == 'DRAW_CARD' and cmd.get('type') == 'DRAW_CARD':
                    cmd['type'] = 'TRANSITION'
                    if 'amount' not in cmd and 'value1' in cmd:
                        cmd['amount'] = cmd.get('value1')
            except Exception:
                pass

            return cmd

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
            cmds.append(wrap_action(a))
        return cmds
    except Exception:
        return []


__all__ = ["ICommand", "BaseCommand", "wrap_action", "generate_legal_commands"]
