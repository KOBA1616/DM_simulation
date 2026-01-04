# -*- coding: utf-8 -*-
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.constants import RESERVED_VARIABLES
from dm_toolkit.gui.editor.forms.action_config import ACTION_UI_CONFIG

class VariableLinkManager:
    """
    Helper class to manage variable linking logic for actions.
    Separates data retrieval and key generation from UI components.
    """

    @staticmethod
    def get_available_variables(current_item):
        """
        Retrieves a list of available variables from preceding siblings in the model.
        Returns a list of tuples: (label, key).
        """
        variables = []

        # Add Manual Input option (represented by empty key)
        variables.append((tr("Manual Input"), ""))

        # Add Reserved Constants
        for key, desc in RESERVED_VARIABLES.items():
            variables.append((f"{key} ({tr(desc)})", key))

        if not current_item:
            return variables

        parent = current_item.parent()
        if not parent:
            return variables

        row = current_item.row()
        for i in range(row):
            sibling = parent.child(i)
            sib_data = sibling.data(Qt.ItemDataRole.UserRole + 2)
            if not sib_data:
                continue

            out_key = sib_data.get('output_value_key')
            if out_key:
                type_disp = tr(sib_data.get('type'))

                # Enhance label with Output Port Name if available
                sib_type = sib_data.get('type')
                sib_config = ACTION_UI_CONFIG.get(sib_type, {})
                outputs = sib_config.get('outputs', {})
                port_name = outputs.get('output_value_key', '')

                if port_name:
                    label = f"Step {i}: {type_disp} -> {tr(port_name)}"
                else:
                    label = f"Step {i}: {type_disp}"

                variables.append((label, out_key))

        return variables

    @staticmethod
    def generate_output_key(current_item, action_type):
        """
        Generates a unique output key for the current item.
        Uses UUID from item data if available, otherwise falls back to row index.
        """
        if not current_item:
            return ""

        action_data = current_item.data(Qt.ItemDataRole.UserRole + 2)
        uid = action_data.get('uid')

        if uid:
            # Use UUID to ensure uniqueness
            return f"var_{uid}"
        else:
            # Fallback to row index if no UUID
            row = current_item.row()
            return f"var_{action_type}_{row}"

    @staticmethod
    def get_input_label(action_type):
        """
        Returns the localized label for the input source field based on the action type.
        """
        config = ACTION_UI_CONFIG.get(action_type, {})
        inputs = config.get('inputs', {})
        if 'input_value_key' in inputs:
            return f"{tr('Input Source')} ({tr(inputs['input_value_key'])})"
        else:
            return tr("Input Source")
