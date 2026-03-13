# -*- coding: utf-8 -*-
"""CommandConverter compatibility wrapper.

Provides `CommandConverter.convert()` as the canonical name for converters.
Internally uses the legacy `ActionConverter` shim when present to preserve
backwards compatibility during migration.
"""
from typing import Any, Dict

try:
    from dm_toolkit.gui.editor.action_converter import ActionConverter
except Exception:
    ActionConverter = None


class CommandConverter:
    @staticmethod
    def convert(action: Any) -> Dict[str, Any]:
        if ActionConverter is not None:
            return ActionConverter.convert(action)
        # Fallback: best-effort mapping
        if isinstance(action, dict):
            return action
        if hasattr(action, 'to_dict'):
            try:
                return action.to_dict()
            except Exception:
                pass
        return {'type': str(getattr(action, 'type', 'UNKNOWN'))}
