from PyQt6.QtWidgets import QWidget, QFormLayout, QCheckBox, QComboBox, QLabel, QLineEdit
from PyQt6.QtCore import pyqtSignal, Qt
from dm_toolkit.gui.localization import tr

class VariableLinkWidget(QWidget):
    """
    Reusable widget for linking variables (Action Chaining).
    Handles 'Smart Link' checkbox, Input Key selection (from previous steps), and Output Key generation.
    """

    # Signal emitted when any property changes
    linkChanged = pyqtSignal()
    # Signal emitted when smart link check state changes (to update parent visibility)
    smartLinkStateChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_item = None

    def setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.smart_link_check = QCheckBox(tr("Use result from previous measurement"))
        layout.addRow(self.smart_link_check)

        self.input_key_combo = QComboBox()
        self.input_key_combo.setEditable(False)
        self.input_key_label = QLabel(tr("Input Key"))
        layout.addRow(self.input_key_label, self.input_key_combo)

        self.output_key_label = QLabel(tr("Output Key"))
        self.output_key_edit = QLineEdit()
        layout.addRow(self.output_key_label, self.output_key_edit)
        self.output_key_label.setVisible(False)
        self.output_key_edit.setVisible(False)

        # Connect signals
        self.smart_link_check.stateChanged.connect(self.on_check_changed)
        self.input_key_combo.currentIndexChanged.connect(self.linkChanged.emit)
        self.output_key_edit.textChanged.connect(self.linkChanged.emit)

        # Initial visibility
        self.update_visibility()

    def set_current_item(self, item):
        self.current_item = item

    def on_check_changed(self):
        is_checked = self.smart_link_check.isChecked()
        self.update_visibility()
        self.smartLinkStateChanged.emit(is_checked)

        # Auto-select last available if checked and nothing selected
        if is_checked and self.input_key_combo.currentIndex() == -1:
             self.populate_input_keys()
             if self.input_key_combo.count() > 0:
                 self.input_key_combo.setCurrentIndex(self.input_key_combo.count() - 1)

        self.linkChanged.emit()

    def update_visibility(self):
        checked = self.smart_link_check.isChecked()
        self.input_key_label.setVisible(checked)
        self.input_key_combo.setVisible(checked)

    def set_data(self, data):
        self.blockSignals(True)

        input_key = data.get('input_value_key', '')
        self.smart_link_check.setChecked(bool(input_key))
        self.output_key_edit.setText(data.get('output_value_key', ''))

        self.populate_input_keys()

        # Match key
        found = False
        for i in range(self.input_key_combo.count()):
             if self.input_key_combo.itemData(i) == input_key:
                  self.input_key_combo.setCurrentIndex(i)
                  found = True
                  break

        if not found and input_key:
             self.input_key_combo.addItem(input_key, input_key)
             self.input_key_combo.setCurrentIndex(self.input_key_combo.count()-1)

        self.update_visibility()
        self.blockSignals(False)

    def get_data(self, data):
        """
        Updates the data dictionary in-place.
        """
        # Input Key
        idx = self.input_key_combo.currentIndex()
        if idx >= 0:
             data['input_value_key'] = self.input_key_combo.itemData(idx)
        else:
             data['input_value_key'] = self.input_key_combo.currentText()

        # If smart link is unchecked, we might want to clear input_value_key?
        if not self.smart_link_check.isChecked():
            data['input_value_key'] = ""

        # Output Key
        out_key = self.output_key_edit.text()
        # Auto-generate if needed is handled by parent or here?
        # Parent knows ActionType, so let's let parent trigger auto-gen or pass info down.
        # But we simply read the edit field here.
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
        self.input_key_combo.clear()
        if not self.current_item: return

        parent = self.current_item.parent()
        if not parent: return

        row = self.current_item.row()
        for i in range(row):
            sibling = parent.child(i)
            sib_data = sibling.data(Qt.ItemDataRole.UserRole + 2)
            out_key = sib_data.get('output_value_key')
            if out_key:
                type_disp = tr(sib_data.get('type'))
                label = f"Step {i}: {type_disp}"
                self.input_key_combo.addItem(label, out_key)

    def is_smart_link_active(self):
        return self.smart_link_check.isChecked()

    def set_smart_link_enabled(self, enabled):
        """
        Controls whether the smart link checkbox is visible/enabled.
        Some actions (like Apply Modifier) depend on it.
        """
        self.smart_link_check.setVisible(enabled)
        # If hidden, ensure related widgets are hidden too (via update_visibility logic or check state)
        if not enabled:
             self.input_key_label.setVisible(False)
             self.input_key_combo.setVisible(False)
