# -*- coding: utf-8 -*-
from PyQt6.QtCore import Qt
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.constants import RESERVED_VARIABLES
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG

class VariableLinkManager:
    """
    Helper class to manage variable linking logic for commands.
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
                # The type stored in data is the command type (e.g. DRAW_CARD)
                sib_type = sib_data.get('type')
                type_disp = tr(sib_type)

                # Enhance label with Output Port Name if available in COMMAND_UI_CONFIG
                sib_config = COMMAND_UI_CONFIG.get(sib_type, {})
                outputs = sib_config.get('outputs', {})
                port_name = outputs.get('output_value_key', '')

                if port_name:
                    label = f"Step {i}: {type_disp} -> {tr(port_name)}"
                else:
                    label = f"Step {i}: {type_disp}"

                variables.append((label, out_key))

        return variables

    @staticmethod
    def generate_output_key(current_item, command_type):
        """
        Generates a unique output key for the current item.
        Uses UUID from item data if available, otherwise falls back to row index.
        """
        if not current_item:
            return ""

        command_data = current_item.data(Qt.ItemDataRole.UserRole + 2)
        uid = command_data.get('uid')

        if uid:
            # Use UUID to ensure uniqueness
            return f"var_{uid}"
        else:
            # Fallback to row index if no UUID
            row = current_item.row()
            return f"var_{command_type}_{row}"

    @staticmethod
    def get_input_label(command_type):
        """
        Returns the localized label for the input source field based on the command type.
        """
        # Note: COMMAND_UI_CONFIG does not currently have explicit 'inputs' config like ACTION_UI_CONFIG did,
        # but we can look for 'input_link' in 'visible' to determine relevance,
        # or just return a generic label.
        # If we want specific labels, we should add 'input_label' to CommandDef in command_config.py

        # For now, return generic label
        return tr("Input Source")
