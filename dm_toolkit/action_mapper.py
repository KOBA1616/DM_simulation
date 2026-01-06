# -*- coding: utf-8 -*-
from typing import Any, Dict
from dm_toolkit.action_to_command import map_action as _map_action

"""
DEPRECATED: Use dm_toolkit.unified_execution instead.

This module provides legacy mapping logic that is being phased out in favor of
the Unified Execution Pipeline.
"""

class ActionToCommandMapper:
    """
    Central logic for converting Action dictionaries to Command dictionaries.
    Wrapper around the pure function implementation in action_to_command.py
    """

    @staticmethod
    def map_action(action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a legacy Action dictionary to a Command dictionary.
        """
        # Strictly delegate to the canonical mapper (Phase 1: Normalization)
        return _map_action(action_data)
