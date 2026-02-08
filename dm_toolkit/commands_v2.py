"""Command-first API shim for Python.

Provides a thin wrapper around native `dm_ai_module.generate_commands` and
optional fallback to legacy generators when `strict` is False.
"""
from typing import Any, List


def generate_legal_commands(state: Any, card_db: Any, strict: bool = False) -> List[Any]:
    try:
        import dm_ai_module
    except Exception:
        if strict:
            raise RuntimeError("Native dm_ai_module not available for generate_commands")
        # Fallback: import legacy helper
        try:
            from dm_toolkit import commands as legacy
            return legacy.generate_legal_commands(state, card_db)
        except Exception:
            return []

    # Prefer top-level generate_commands
    if hasattr(dm_ai_module, 'generate_commands'):
        try:
            return dm_ai_module.generate_commands(state, card_db) or []
        except Exception:
            if strict:
                raise

    # Fallback to legacy Intent/Action generators when not strict
    if not strict:
        try:
            from dm_toolkit import commands as legacy
            return legacy.generate_legal_commands(state, card_db)
        except Exception:
            return []

    raise RuntimeError("No command generator available (strict mode)")
