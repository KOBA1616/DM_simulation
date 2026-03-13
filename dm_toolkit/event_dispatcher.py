from typing import Callable, Dict, List, Any, Tuple
import threading
import uuid

Callback = Callable[[Any], None]


class EventDispatcher:
    """Simple thread-safe event dispatcher for Python-side consumers.

    Usage:
        ed = EventDispatcher()
        token = ed.subscribe('STATE_CHANGED', lambda ev: print(ev))
        ed.emit('STATE_CHANGED', {'state': 'started'})
        ed.unsubscribe(token)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # mapping: event_type -> list of (token, callback)
        self._subs: Dict[str, List[Tuple[str, Callback]]] = {}

    def subscribe(self, event_type: str, callback: Callback) -> str:
        """Subscribe to an event. Returns a token for unsubscription."""
        token = str(uuid.uuid4())
        with self._lock:
            self._subs.setdefault(event_type, []).append((token, callback))
        return token

    def unsubscribe(self, token: str) -> bool:
        """Unsubscribe by token. Returns True if removed."""
        with self._lock:
            for etype, lst in list(self._subs.items()):
                for i, (t, cb) in enumerate(lst):
                    if t == token:
                        lst.pop(i)
                        if not lst:
                            del self._subs[etype]
                        return True
        return False

    def emit(self, event_type: str, event: Any) -> None:
        """Emit an event synchronously to all subscribers for event_type."""
        # Make a snapshot of callbacks under lock, call outside lock
        with self._lock:
            listeners = list(self._subs.get(event_type, []))
        for _token, cb in listeners:
            try:
                cb(event)
            except Exception:
                # Swallow exceptions from user callbacks to avoid breaking emitter
                # Real implementation should log the exception.
                pass
