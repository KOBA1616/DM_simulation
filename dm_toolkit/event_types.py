"""Common event type constants for Python-side event system.

These string constants are used with `EventDispatcher.subscribe` / `emit`.
They mirror the canonical event names that will be exposed by native bindings
in a later step.
"""

STATE_CHANGED = "STATE_CHANGED"
ACTION_EXECUTED = "ACTION_EXECUTED"
COMMAND_EMITTED = "COMMAND_EMITTED"
TURN_STARTED = "TURN_STARTED"
TURN_ENDED = "TURN_ENDED"

__all__ = [
    'STATE_CHANGED',
    'ACTION_EXECUTED',
    'COMMAND_EMITTED',
    'TURN_STARTED',
    'TURN_ENDED',
]
