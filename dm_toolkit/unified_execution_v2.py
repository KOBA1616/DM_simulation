"""Minimal command-first execution helpers.

These helpers expect `CommandDef` objects (from native bindings) and delegate
execution to `EngineCompat.ExecuteCommand` which understands the engine format.
"""
from typing import Any, Iterable


def execute_command(state: Any, command: Any, card_db: Any = None) -> None:
    from dm_toolkit.engine.compat import EngineCompat
    # EngineCompat.ExecuteCommand accepts dicts or bound CommandDef objects
    if card_db is not None:
        EngineCompat.ExecuteCommand(state, command, card_db)
    else:
        EngineCompat.ExecuteCommand(state, command)


def execute_commands(state: Any, commands: Iterable[Any], card_db: Any = None) -> None:
    for c in commands:
        execute_command(state, c, card_db)


def to_command_dict(command: Any) -> dict:
    # Prefer native to_dict if present (pybind-exposed), else fall back to existing
    # unified_execution.to_command_dict for compatibility.
    try:
        if hasattr(command, 'to_dict') and callable(getattr(command, 'to_dict')):
            return command.to_dict()
    except Exception:
        pass
    try:
        from dm_toolkit.unified_execution import to_command_dict as _to_dict
        return _to_dict(command)
    except Exception:
        return {}
