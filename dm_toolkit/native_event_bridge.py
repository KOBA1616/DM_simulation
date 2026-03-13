from typing import Any, Callable
from .event_dispatcher import EventDispatcher

# Singleton dispatcher for bridging native events into Python subscribers
_dispatcher = EventDispatcher()


def subscribe(event_type: str, callback: Callable[[Any], None]) -> str:
    """Subscribe to events forwarded from native code.

Native code will call `native_emit(event_type, payload)` (via pybind11)
and this module will forward to Python subscribers.
"""
    return _dispatcher.subscribe(event_type, callback)


def unsubscribe(token: str) -> bool:
    return _dispatcher.unsubscribe(token)


def native_emit(event_type: str, payload: Any) -> None:
    """Called by native bindings to emit an event into Python.

This function is intended to be a small stable API surface that C++
can call via pybind11 (e.g., `dm.native_emit(type_str, dict)`).
"""
    _dispatcher.emit(event_type, payload)


def get_dispatcher() -> EventDispatcher:
    """Return the singleton EventDispatcher (for advanced usage)."""
    return _dispatcher
