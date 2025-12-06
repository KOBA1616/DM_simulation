from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox, QGroupBox, QGridLayout, QCheckBox, QSpinBox, QLabel, QLineEdit
from PyQt6.QtCore import Qt
from gui.localization import tr

# Configuration for Condition UI logic
CONDITION_UI_CONFIG = {
    "NONE": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "MANA_ARMED": {
        "show_val": True,
        "show_str": True, # Usually specifies civ
        "label_val": "Count",
        "label_str": "Civilization"
    },
    "SHIELD_COUNT": {
        "show_val": True,
        "show_str": False,
        "label_val": "Count",
        "label_str": "Comparison (Optional)"
    },
    "CIVILIZATION_MATCH": {
        "show_val": False,
        "show_str": True,
        "label_val": "Value",
        "label_str": "Civilization"
    },
    "OPPONENT_PLAYED_WITHOUT_MANA": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    }
}

class EffectEditForm(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.trigger_combo = QComboBox()
        triggers = [
            "ON_PLAY", "ON_ATTACK", "ON_DESTROY", "S_TRIGGER", "TURN_START", "PASSIVE_CONST", "ON_OTHER_ENTER",
            "ON_ATTACK_FROM_HAND"
        ]
        for t in triggers:
            self.trigger_combo.addItem(tr(t), t)
        layout.addRow(tr("Trigger"), self.trigger_combo)

        # Condition (Simplified)
        self.condition_group = QGroupBox(tr("Condition"))
        c_layout = QGridLayout(self.condition_group)
        self.cond_type_combo = QComboBox()
        cond_types = ["NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH", "OPPONENT_PLAYED_WITHOUT_MANA"]
        for c in cond_types:
            self.cond_type_combo.addItem(tr(c), c)

        c_layout.addWidget(QLabel(tr("Type")), 0, 0)
        c_layout.addWidget(self.cond_type_combo, 0, 1)

        # Value Row
        self.lbl_val = QLabel(tr("Value"))
        self.cond_val_spin = QSpinBox()
        c_layout.addWidget(self.lbl_val, 1, 0)
        c_layout.addWidget(self.cond_val_spin, 1, 1)

        # String Row
        self.lbl_str = QLabel(tr("String Value"))
        self.cond_str_edit = QLineEdit()
        c_layout.addWidget(self.lbl_str, 2, 0)
        c_layout.addWidget(self.cond_str_edit, 2, 1)

        layout.addRow(self.condition_group)

        # Connect signals
        self.trigger_combo.currentIndexChanged.connect(self.update_data)
        self.cond_type_combo.currentIndexChanged.connect(self.on_cond_type_changed)
        self.cond_val_spin.valueChanged.connect(self.update_data)
        self.cond_str_edit.textChanged.connect(self.update_data)

        # Initial UI State
        self.update_ui_visibility("NONE")

    def on_cond_type_changed(self):
        ctype = self.cond_type_combo.currentData()
        self.update_ui_visibility(ctype)
        self.update_data()

    def update_ui_visibility(self, condition_type):
        config = CONDITION_UI_CONFIG.get(condition_type, CONDITION_UI_CONFIG["NONE"])

        show_val = config.get("show_val", True)
        label_val = config.get("label_val", "Value")

        show_str = config.get("show_str", True)
        label_str = config.get("label_str", "String Value")

        self.lbl_val.setText(tr(label_val))
        self.lbl_val.setVisible(show_val)
        self.cond_val_spin.setVisible(show_val)

        self.lbl_str.setText(tr(label_str))
        self.lbl_str.setVisible(show_str)
        self.cond_str_edit.setVisible(show_str)

    def set_data(self, item):
        self.current_item = item
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.block_signals(True)

        trig_idx = self.trigger_combo.findData(data.get('trigger', 'ON_PLAY'))
        if trig_idx >= 0:
            self.trigger_combo.setCurrentIndex(trig_idx)

        cond = data.get('condition', {})
        ctype = cond.get('type', 'NONE')
        cond_idx = self.cond_type_combo.findData(ctype)
        if cond_idx >= 0:
            self.cond_type_combo.setCurrentIndex(cond_idx)

        self.cond_val_spin.setValue(cond.get('value', 0))
        self.cond_str_edit.setText(cond.get('str_val', ''))

        self.update_ui_visibility(ctype)

        self.block_signals(False)

    def update_data(self):
        if not self.current_item: return
        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)

        data['trigger'] = self.trigger_combo.currentData()

        cond = {}
        cond['type'] = self.cond_type_combo.currentData()
        cond['value'] = self.cond_val_spin.value()
        str_val = self.cond_str_edit.text()
        if str_val: cond['str_val'] = str_val
        data['condition'] = cond

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        # We need to translate the trigger back for display or just use the Japanese text?
        # LogicTreeWidget handles the display text update usually, but here we update it explicitly.
        # Let's use tr() on the data value.
        self.current_item.setText(f"{tr('Effect')}: {tr(data['trigger'])}")

    def block_signals(self, block):
        self.trigger_combo.blockSignals(block)
        self.cond_type_combo.blockSignals(block)
        self.cond_val_spin.blockSignals(block)
        self.cond_str_edit.blockSignals(block)
