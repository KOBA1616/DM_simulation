# -*- coding: utf-8 -*-
"""CommandConverter compatibility wrapper.

Provides `CommandConverter.convert()` as the canonical name for converters.
Internally uses the legacy `ActionConverter` shim when present to preserve
backwards compatibility during migration.
"""
from typing import Any, Dict


class CommandConverter:
    @staticmethod
    def convert(action: Any) -> Dict[str, Any]:
        """Convert various action-like inputs to a normalized command dict.

        This implementation is self-contained and no longer depends on the
        legacy `action_converter` shim. It accepts dicts, objects exposing
        `to_dict()`, or falls back to a minimal representation.
        """
        if isinstance(action, dict):
            return action
        if hasattr(action, 'to_dict'):
            try:
                return action.to_dict()
            except Exception:
                pass
        return {'type': str(getattr(action, 'type', 'UNKNOWN'))}
