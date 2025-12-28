# -*- coding: utf-8 -*-
from dm_toolkit.action_to_command import ActionToCommand

class ActionToCommandMapper:
    """
    Central logic for converting Action dictionaries to Command dictionaries.
    Delegates to the new standardized ActionToCommand implementation.
    """

    @staticmethod
    def map_action(action_data):
        """
        Converts a legacy Action dictionary to a Command dictionary complying with the Schema.
        """
        return ActionToCommand.map_action(action_data)
