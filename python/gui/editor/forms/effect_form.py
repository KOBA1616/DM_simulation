from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QSpinBox, QLineEdit, QLabel
from PyQt6.QtCore import Qt
from gui.localization import tr

class EffectEditForm(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.trigger_combo = QComboBox()
        self.trigger_combo.addItems(["ON_PLAY", "ON_ATTACK", "ON_DESTROY", "PASSIVE_CONST", "TURN_START", "ON_OTHER_ENTER", "ON_ATTACK_FROM_HAND", "S_TRIGGER", "NONE"])
        layout.addRow(tr("Trigger"), self.trigger_combo)

        self.cond_type_combo = QComboBox()
        self.cond_type_combo.addItems(["NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH", "OPPONENT_PLAYED_WITHOUT_MANA"])
        layout.addRow(tr("Condition Type"), self.cond_type_combo)

        self.cond_val_spin = QSpinBox()
        layout.addRow(tr("Condition Value"), self.cond_val_spin)

        self.cond_str_edit = QLineEdit()
        layout.addRow(tr("Condition String"), self.cond_str_edit)

        self.trigger_combo.currentTextChanged.connect(self.update_data)
        self.cond_type_combo.currentTextChanged.connect(self.update_data)
        self.cond_val_spin.valueChanged.connect(self.update_data)
        self.cond_str_edit.textChanged.connect(self.update_data)

    def set_data(self, item):
        self.current_item = item
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.block_signals(True)
        self.trigger_combo.setCurrentText(data.get('trigger', 'NONE'))
        cond = data.get('condition', {})
        self.cond_type_combo.setCurrentText(cond.get('type', 'NONE'))
        self.cond_val_spin.setValue(cond.get('value', 0))
        self.cond_str_edit.setText(cond.get('str_val', ''))
        self.block_signals(False)

    def update_data(self):
        if not self.current_item: return
        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)

        data['trigger'] = self.trigger_combo.currentText()
        data['condition'] = {
            'type': self.cond_type_combo.currentText(),
            'value': self.cond_val_spin.value(),
            'str_val': self.cond_str_edit.text()
        }

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        self.current_item.setText(f"Effect: {data['trigger']}")

    def block_signals(self, block):
        self.trigger_combo.blockSignals(block)
        self.cond_type_combo.blockSignals(block)
        self.cond_val_spin.blockSignals(block)
        self.cond_str_edit.blockSignals(block)
