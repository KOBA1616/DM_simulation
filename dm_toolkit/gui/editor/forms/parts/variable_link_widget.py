from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QLabel, QLineEdit
from PyQt6.QtCore import pyqtSignal, Qt
from dm_toolkit.gui.localization import tr

class VariableLinkWidget(QWidget):
    """
    Reusable widget for linking variables (Action Chaining).
    Handles 'Input Source' selection (Manual, Event Source, or Previous Steps) and Output Key generation.
    """

    # Signal emitted when any property changes
    linkChanged = pyqtSignal()
    # Signal emitted when smart link check state changes (to update parent visibility)
    # Kept for compatibility, mapped to "is input source not manual"
    smartLinkStateChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Replaced Checkbox with specific Combo
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
        self.input_key_combo.currentIndexChanged.connect(self.on_input_changed)
        self.output_key_edit.textChanged.connect(self.linkChanged.emit)

        # Initial Population
        self.populate_input_keys()

    def set_current_item(self, item):
        self.current_item = item
        self.populate_input_keys()

    def on_input_changed(self):
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

        # Fallback for "Manual" (empty string)
        if not found and input_key == "":
             # Should be index 0
             self.input_key_combo.setCurrentIndex(0)
             found = True

        if not found and input_key:
             # Add unknown key temporarily? Or just set manual and warn?
             # For now, append it so we don't lose data
             self.input_key_combo.addItem(f"{input_key} (Unknown)", input_key)
             self.input_key_combo.setCurrentIndex(self.input_key_combo.count()-1)

        self.blockSignals(False)
        # Emit state change to ensure parent UI updates (hiding val1 etc)
        self.smartLinkStateChanged.emit(self.is_smart_link_active())

    def get_data(self, data):
        """
        Updates the data dictionary in-place.
        """
        # Input Key
        idx = self.input_key_combo.currentIndex()
        if idx >= 0:
             val = self.input_key_combo.itemData(idx)
             data['input_value_key'] = val if val is not None else ""
        else:
             data['input_value_key'] = ""

        # Output Key
        out_key = self.output_key_edit.text()
        data['output_value_key'] = out_key

    def ensure_output_key(self, action_type, produces_output):
        """
        Generates output key if missing and required.
        """
        if produces_output and not self.output_key_edit.text() and self.current_item:
             row = self.current_item.row()
             new_key = f"var_{action_type}_{row}"
             self.output_key_edit.setText(new_key)
             # This triggers textChanged -> linkChanged -> update_data

    def populate_input_keys(self):
        current_data = self.input_key_combo.currentData()
        self.input_key_combo.clear()

        # 1. Manual Input
        self.input_key_combo.addItem(tr("Manual Value / None"), "")

        # 2. Event Source
        self.input_key_combo.addItem(tr("Event Source"), "EVENT_SOURCE")

        # 3. Dynamic Keys from siblings
        if self.current_item:
            parent = self.current_item.parent()
            if parent:
                row = self.current_item.row()
                for i in range(row):
                    sibling = parent.child(i)
                    sib_data = sibling.data(Qt.ItemDataRole.UserRole + 2)
                    if not sib_data:
                        continue
                    out_key = sib_data.get('output_value_key')
                    if out_key:
                        type_disp = tr(sib_data.get('type'))
                        label = f"Step {i}: {type_disp}"
                        self.input_key_combo.addItem(label, out_key)

        # Restore selection if possible
        if current_data is not None:
             idx = self.input_key_combo.findData(current_data)
             if idx >= 0:
                  self.input_key_combo.setCurrentIndex(idx)
             else:
                  self.input_key_combo.setCurrentIndex(0)

    def is_smart_link_active(self):
        # Active if selected data is NOT empty string
        val = self.input_key_combo.currentData()
        return bool(val)

    def set_smart_link_enabled(self, enabled):
        """
        Controls whether the input source combo is visible/enabled.
        """
        self.input_key_label.setVisible(enabled)
        self.input_key_combo.setVisible(enabled)

        if not enabled:
             # Reset to Manual if disabled to avoid hidden state affecting logic
             self.input_key_combo.setCurrentIndex(0)
