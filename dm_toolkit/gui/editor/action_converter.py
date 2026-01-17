# -*- coding: utf-8 -*-
"""Legacy ActionConverter compatibility shim.

This project standardizes Action->Command conversion in `dm_toolkit.action_to_command.map_action`
(see AGENTS.md). Older tests and integrations may still import
`dm_toolkit.gui.editor.action_converter.ActionConverter`.

This module intentionally contains *no mapping logic* and delegates to `map_action`.
"""

from __future__ import annotations

from typing import Any, Dict

from dm_toolkit.action_to_command import map_action


class ActionConverter:
    @staticmethod
    def convert(action: Any) -> Dict[str, Any]:
        return map_action(action)
