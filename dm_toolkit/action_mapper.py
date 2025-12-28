# -*- coding: utf-8 -*-
from typing import Any, Dict
from dm_toolkit.action_to_command import map_action as _map_action

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
        return _map_action(action_data)
