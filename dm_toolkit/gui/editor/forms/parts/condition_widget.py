# -*- coding: cp932 -*-
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QComboBox, QSpinBox, QLineEdit, QLabel, QGroupBox
)
from PyQt6.QtCore import pyqtSignal
from dm_toolkit.gui.i18n import tr
from typing import Any, cast
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget

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
    },
    "OPPONENT_DRAW_COUNT": {
        "show_val": True,
        "show_str": False,
        "label_val": "Count (>=)",
        "label_str": "String"
    },
    "DURING_YOUR_TURN": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "DURING_OPPONENT_TURN": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "FIRST_ATTACK": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "EVENT_FILTER_MATCH": {
        "show_val": False,
        "show_str": False,
        "show_filter": True
    },
    # Expanded Types
    "COMPARE_STAT": {
        "show_val": True,
        "show_stat_key": True,
        "show_op": True,
        "label_val": "Threshold",
        "label_stat_key": "Stat Key (e.g. MY_SHIELD_COUNT)",
        "label_op": "Operator"
    },
    "CARDS_MATCHING_FILTER": {
        "show_val": True,
        "show_op": True,
        "show_filter": True,
        "label_val": "Count Threshold",
        "label_op": "Operator"
    },
    "DECK_EMPTY": {
        "show_val": False,
        "show_str": False
    },
    "CUSTOM": {
        "show_type_edit": True,
        "show_val": True,
        "show_str": True,
        "show_stat_key": True,
        "show_op": True,
        "show_filter": True,
        "label_val": "Value",
        "label_str": "String",
        "label_stat_key": "Stat Key",
        "label_op": "Operator"
    }
}

class ConditionEditorWidget(QGroupBox):
    dataChanged = pyqtSignal()

    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        self.setTitle(title if title else tr("Condition"))
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)

        # Type Row
        self.cond_type_combo = QComboBox()
        cond_types = [
            "NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH",
            "OPPONENT_PLAYED_WITHOUT_MANA", "OPPONENT_DRAW_COUNT",
            "DURING_YOUR_TURN", "DURING_OPPONENT_TURN",
            "FIRST_ATTACK", "EVENT_FILTER_MATCH",
            "COMPARE_STAT", "CARDS_MATCHING_FILTER", "DECK_EMPTY",
            "CUSTOM"
        ]
        self.populate_combo(self.cond_type_combo, cond_types)
        self.cond_type_combo.currentIndexChanged.connect(self.on_cond_type_changed)
        self.cond_type_combo.currentIndexChanged.connect(self.dataChanged.emit)

        layout.addWidget(QLabel(tr("Type")), 0, 0)
        layout.addWidget(self.cond_type_combo, 0, 1)

        # Custom Type Edit (Row 1)
        self.lbl_type_edit = QLabel(tr("Custom Type"))
        self.type_edit = QLineEdit()
        self.type_edit.textChanged.connect(self.dataChanged.emit)
        layout.addWidget(self.lbl_type_edit, 1, 0)
        layout.addWidget(self.type_edit, 1, 1)

        # Stat Key (Row 2)
        self.lbl_stat_key = QLabel(tr("Stat Key"))
        self.stat_key_combo = QComboBox()
        self.stat_key_combo.setEditable(True)
        common_stats = [
            "MY_MANA_COUNT", "OPPONENT_MANA_COUNT",
            "MY_HAND_COUNT", "OPPONENT_HAND_COUNT",
            "MY_SHIELD_COUNT", "OPPONENT_SHIELD_COUNT",
            "MY_BATTLE_ZONE_COUNT", "OPPONENT_BATTLE_ZONE_COUNT"
        ]
        self.populate_combo(self.stat_key_combo, common_stats)
        self.stat_key_combo.editTextChanged.connect(self.dataChanged.emit)
        self.stat_key_combo.currentIndexChanged.connect(self.dataChanged.emit)
        layout.addWidget(self.lbl_stat_key, 2, 0)
        layout.addWidget(self.stat_key_combo, 2, 1)

        # Operator (Row 3)
        self.lbl_op = QLabel(tr("Operator"))
        self.op_combo = QComboBox()
        ops = [">", "<", "=", ">=", "<=", "!="]
        self.populate_combo(self.op_combo, ops)
        self.op_combo.currentTextChanged.connect(self.dataChanged.emit)
        layout.addWidget(self.lbl_op, 3, 0)
        layout.addWidget(self.op_combo, 3, 1)

        # Value Row (Row 4)
        self.lbl_val = QLabel(tr("Value"))
        self.cond_val_spin = QSpinBox()
        self.cond_val_spin.setRange(-9999, 9999)
        self.cond_val_spin.valueChanged.connect(self.dataChanged.emit)

        layout.addWidget(self.lbl_val, 4, 0)
        layout.addWidget(self.cond_val_spin, 4, 1)

        # String Row (Row 5)
        self.lbl_str = QLabel(tr("String Value"))
        self.cond_str_edit = QLineEdit()
        self.cond_str_edit.textChanged.connect(self.dataChanged.emit)

        layout.addWidget(self.lbl_str, 5, 0)
        layout.addWidget(self.cond_str_edit, 5, 1)

        # Filter Widget (Row 6)
        self.cond_filter = FilterEditorWidget()
        self.cond_filter.filterChanged.connect(self.dataChanged.emit)
        self.cond_filter.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})
        self.cond_filter.setVisible(False)
        layout.addWidget(self.cond_filter, 6, 0, 1, 2)

        # Initial Update
        self.update_ui_visibility("NONE")

    def populate_combo(self, combo, items):
        combo.clear()
        for item in items:
            combo.addItem(tr(str(item)), str(item))

    def on_cond_type_changed(self):
        ctype = self.cond_type_combo.currentData()
        self.update_ui_visibility(ctype)
        # dataChanged is connected directly

    def update_ui_visibility(self, condition_type):
        config = cast(dict[str, Any], CONDITION_UI_CONFIG.get(condition_type, CONDITION_UI_CONFIG["NONE"]))
        # Fallback to CUSTOM if type not found (should be handled by setting combo to CUSTOM manually)
        if condition_type == "CUSTOM":
            config = CONDITION_UI_CONFIG["CUSTOM"]

        show_type_edit = config.get("show_type_edit", False)

        show_val = config.get("show_val", True)
        label_val = config.get("label_val", "Value")

        show_str = config.get("show_str", True)
        label_str = config.get("label_str", "String Value")

        show_stat_key = config.get("show_stat_key", False)
        label_stat_key = config.get("label_stat_key", "Stat Key")

        show_op = config.get("show_op", False)
        label_op = config.get("label_op", "Operator")

        show_filter = config.get("show_filter", False)

        self.lbl_type_edit.setVisible(show_type_edit)
        self.type_edit.setVisible(show_type_edit)

        self.lbl_val.setText(tr(label_val))
        self.lbl_val.setVisible(show_val)
        self.cond_val_spin.setVisible(show_val)

        self.lbl_str.setText(tr(label_str))
        self.lbl_str.setVisible(show_str)
        self.cond_str_edit.setVisible(show_str)

        self.lbl_stat_key.setText(tr(label_stat_key))
        self.lbl_stat_key.setVisible(show_stat_key)
        self.stat_key_combo.setVisible(show_stat_key)

        self.lbl_op.setText(tr(label_op))
        self.lbl_op.setVisible(show_op)
        self.op_combo.setVisible(show_op)

        self.cond_filter.setVisible(show_filter)

    def set_data(self, data):
        self.blockSignals(True)

        ctype = data.get('type', 'NONE')

        # Check if known type
        idx = self.cond_type_combo.findData(ctype)
        if idx >= 0:
            self.cond_type_combo.setCurrentIndex(idx)
            self.type_edit.clear()
        else:
            # Custom type
            custom_idx = self.cond_type_combo.findData("CUSTOM")
            if custom_idx >= 0:
                self.cond_type_combo.setCurrentIndex(custom_idx)
            self.type_edit.setText(ctype)

        value = data.get('value', 0)
        if value is None:
            value = 0
        self.cond_val_spin.setValue(value)
        self.cond_str_edit.setText(data.get('str_val', ''))

        stat_key = data.get('stat_key', '')
        idx_stat = self.stat_key_combo.findText(stat_key)
        if idx_stat >= 0:
            self.stat_key_combo.setCurrentIndex(idx_stat)
        else:
            self.stat_key_combo.setCurrentText(stat_key)

        op = data.get('op', '>')
        idx_op = self.op_combo.findText(op)
        if idx_op >= 0:
            self.op_combo.setCurrentIndex(idx_op)
        else:
            self.op_combo.setCurrentText(op)

        self.cond_filter.set_data(data.get('filter', {}))

        # Refresh visibility based on current selection
        current_selection = self.cond_type_combo.currentData()
        self.update_ui_visibility(current_selection)

        self.blockSignals(False)

    def get_data(self):
        ctype = self.cond_type_combo.currentData()
        if ctype is None:
            ctype = "NONE"

        if ctype == "CUSTOM":
            raw_type = self.type_edit.text().strip()
            if raw_type:
                ctype = raw_type

        data = {
            "type": ctype,
            "value": self.cond_val_spin.value()
        }

        # Include other fields if visible or generic
        combo_selection = self.cond_type_combo.currentData()
        config = cast(dict[str, Any], CONDITION_UI_CONFIG.get(combo_selection, CONDITION_UI_CONFIG["CUSTOM"]))

        if config.get("show_str", False) or combo_selection == "CUSTOM":
            str_val = self.cond_str_edit.text()
            if str_val:
                data['str_val'] = str_val

        if config.get("show_stat_key", False) or combo_selection == "CUSTOM":
            stat_key = self.stat_key_combo.currentText()
            if stat_key:
                data['stat_key'] = stat_key

        if config.get("show_op", False) or combo_selection == "CUSTOM":
            op = self.op_combo.currentText()
            if op:
                data['op'] = op

        if (config.get("show_filter", False) or combo_selection == "CUSTOM") and self.cond_filter.isVisible():
            data['filter'] = self.cond_filter.get_data()

        return data

    def blockSignals(self, block):
        super().blockSignals(block)
        self.cond_type_combo.blockSignals(block)
        self.type_edit.blockSignals(block)
        self.cond_val_spin.blockSignals(block)
        self.cond_str_edit.blockSignals(block)
        self.stat_key_combo.blockSignals(block)
        self.op_combo.blockSignals(block)
        self.cond_filter.blockSignals(block)
