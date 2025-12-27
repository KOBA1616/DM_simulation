from typing import Any, Dict, Optional, Protocol, runtime_checkable


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
        return {"kind": self.__class__.__name__}


def wrap_action(action: Any) -> Optional[ICommand]:
    """Return an `ICommand`-like object for the provided `action`.

    - If `action` already implements `execute`, return it.
    - Otherwise, try to use the compatibility shim in `dm_ai_module.action_to_command`.
    - Fallback to a thin wrapper delegating to `action.execute(state, db=None)` when possible.
    """
    if action is None:
        return None

    # If it's already command-like, return as-is
    if hasattr(action, "execute") and callable(getattr(action, "execute")):
        return action  # type: ignore

    # Try to delegate to dm_ai_module.action_to_command if available
    try:
        import dm_ai_module

        if hasattr(dm_ai_module, "action_to_command"):
            return dm_ai_module.action_to_command(action)
    except Exception:
        pass

    # Fallback wrapper
    class _ActionFallback(BaseCommand):
        def __init__(self, a: Any):
            self._action = a

        def execute(self, state: Any) -> Optional[Any]:
            # Some Action implementations expect (state, db) signature
            try:
                if getattr(self._action, "execute", None):
                    try:
                        return self._action.execute(state)
                    except TypeError:
                        # try (state, db)
                        try:
                            return self._action.execute(state, None)
                        except Exception:
                            return None
            except Exception:
                return None
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
            return {"wrapped_action": str(getattr(self._action, "type", None))}

    return _ActionFallback(action)


__all__ = ["ICommand", "BaseCommand", "wrap_action"]
