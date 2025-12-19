from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QLabel, QLineEdit
from PyQt6.QtCore import pyqtSignal, Qt
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.constants import RESERVED_VARIABLES
from dm_toolkit.gui.editor.forms.action_config import ACTION_UI_CONFIG

class VariableLinkWidget(QWidget):
    """
    Reusable widget for linking variables (Action Chaining).
    Handles 'Input Source' selection (Manual, Event Source, Previous Steps) and Output Key generation.
    """

    # Signal emitted when any property changes
    linkChanged = pyqtSignal()
    # Signal emitted when smart link state changes (to update parent visibility)
    smartLinkStateChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.input_key_combo = QComboBox()
        self.input_key_combo.setEditable(False)
        self.input_key_label = QLabel(tr("Input Source"))
        layout.addRow(self.input_key_label, self.input_key_combo)

        self.output_key_label = QLabel(tr("Output Key"))
        self.output_key_edit = QLineEdit()
        layout.addRow(self.output_key_label, self.output_key_edit)
        self.output_key_label.setVisible(False)
        self.output_key_edit.setVisible(False)

        # Connect signals
        self.input_key_combo.currentIndexChanged.connect(self.on_combo_changed)
        self.output_key_edit.textChanged.connect(self.linkChanged.emit)

        # Initial Population
        self.populate_input_keys()

    def set_current_item(self, item):
        self.current_item = item

        # Update Input Label if defined in config
        action_data = item.data(Qt.ItemDataRole.UserRole + 2)
        if action_data:
            act_type = action_data.get('type')
            config = ACTION_UI_CONFIG.get(act_type, {})
            inputs = config.get('inputs', {})
            if 'input_value_key' in inputs:
                self.input_key_label.setText(f"{tr('Input Source')} ({tr(inputs['input_value_key'])})")
            else:
                self.input_key_label.setText(tr("Input Source"))

        self.populate_input_keys()

    def on_combo_changed(self):
        self.linkChanged.emit()
        self.smartLinkStateChanged.emit(self.is_smart_link_active())

    def set_data(self, data):
        self.blockSignals(True)

        input_key = data.get('input_value_key', '')
        self.output_key_edit.setText(data.get('output_value_key', ''))

        self.populate_input_keys()

        # Match key
        found = False
        # Try to find exact match
        for i in range(self.input_key_combo.count()):
             if self.input_key_combo.itemData(i) == input_key:
                  self.input_key_combo.setCurrentIndex(i)
                  found = True
                  break

        # If not found and not empty, it might be a custom key from JSON
        # If it's manual (empty), index 0 is already Manual
        if not found and input_key:
             # Add it temporarily so we don't lose data
             self.input_key_combo.addItem(f"{input_key} (Unknown)", input_key)
             self.input_key_combo.setCurrentIndex(self.input_key_combo.count()-1)
        elif not found and not input_key:
             self.input_key_combo.setCurrentIndex(0) # Manual Input

        self.blockSignals(False)
        # Emit state change to ensure parent UI updates (hiding val1 etc)
        self.smartLinkStateChanged.emit(self.is_smart_link_active())

    def get_data(self, data):
        """
        Updates the data dictionary in-place.
        """
        # Input Key
        idx = self.input_key_combo.currentIndex()
        val = ""
        if idx >= 0:
             val = self.input_key_combo.itemData(idx)
             if val is None: val = "" # Safety

        data['input_value_key'] = val

        # Output Key
        data['output_value_key'] = self.output_key_edit.text()

    def ensure_output_key(self, action_type, produces_output):
        """
        Generates output key if missing and required.
        Uses UUID if available, otherwise falls back to row (legacy).
        """
        if produces_output and not self.output_key_edit.text() and self.current_item:
             # Try to get UUID from item data
             action_data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)
             uid = action_data.get('uid')

             if uid:
                 # Generate a shorter hash/segment for the key for readability, or use full UUID?
                 # Full UUID is safe.
                 # To keep it friendly, maybe last 8 chars? But collision risk exists.
                 # Let's use `var_{uuid}`
                 new_key = f"var_{uid}"
             else:
                 # Fallback to row index if no UUID
                 row = self.current_item.row()
                 new_key = f"var_{action_type}_{row}"

             self.output_key_edit.setText(new_key)
             # This triggers textChanged -> linkChanged

    def populate_input_keys(self):
        current_data = self.input_key_combo.currentData()
        self.input_key_combo.clear()

        # Default options
        self.input_key_combo.addItem(tr("Manual Input"), "")

        # Reserved Constants
        for key, desc in RESERVED_VARIABLES.items():
            self.input_key_combo.addItem(f"{key} ({tr(desc)})", key)

        if not self.current_item: return

        parent = self.current_item.parent()
        if not parent: return

        row = self.current_item.row()
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

                self.input_key_combo.addItem(label, out_key)

        # Try to restore selection if possible
        if current_data is not None:
             for i in range(self.input_key_combo.count()):
                  if self.input_key_combo.itemData(i) == current_data:
                       self.input_key_combo.setCurrentIndex(i)
                       break

    def is_smart_link_active(self):
        # Active if selected item data is not empty string
        idx = self.input_key_combo.currentIndex()
        if idx < 0: return False
        data = self.input_key_combo.itemData(idx)
        return bool(data)

    def set_smart_link_enabled(self, enabled):
        """
        Controls whether the input source can be changed.
        If disabled, force Manual Input? Or just hide?
        Usually this is called to enable/disable the 'feature'.
        If 'enabled' is False, it means this action does not support input linking or Val1 is hidden.
        """
        self.setVisible(enabled)
        pass
