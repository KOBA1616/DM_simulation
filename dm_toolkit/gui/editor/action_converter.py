# -*- coding: utf-8 -*-
"""Deprecated legacy shim for action->command conversion.

This module is retained only for backward compatibility with older tests
and integrations. It delegates to `CommandConverter` in
`dm_toolkit.gui.editor.command_converter` and emits a DeprecationWarning.
Remove this shim only after all consumers migrate to `CommandConverter`.
"""

from typing import Any, Dict
import warnings

try:
    from dm_toolkit.gui.editor.command_converter import CommandConverter
except Exception:
    CommandConverter = None


class ActionConverter:
    @staticmethod
    def convert(action: Any) -> Dict[str, Any]:
        warnings.warn(
            "dm_toolkit.gui.editor.action_converter is deprecated; use command_converter.CommandConverter.convert",
            DeprecationWarning,
            stacklevel=2,
        )
        if CommandConverter is not None:
            return CommandConverter.convert(action)
        # Fallback: best-effort mapping
        if isinstance(action, dict):
            return action
        if hasattr(action, 'to_dict'):
            try:
                return action.to_dict()
            except Exception:
                pass
        return {'type': str(getattr(action, 'type', 'UNKNOWN'))}
