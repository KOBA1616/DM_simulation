# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QComboBox, QSpinBox, QLineEdit, QLabel, QGroupBox
)
from PyQt6.QtCore import pyqtSignal
from dm_toolkit.gui.localization import tr
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
    "OPTIONAL_EFFECT_EXECUTED": {
        "show_val": False,
        "show_str": False,
        "label_str": "Optional"
    },
    "INPUT_VALUE_MATCH": {
        "show_val": True,
        "show_str": False,
        "label_val": "Expected Value"
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

        # Type
        self.cond_type_combo = QComboBox()
        cond_types = [
            "NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH",
            "OPPONENT_PLAYED_WITHOUT_MANA", "OPPONENT_DRAW_COUNT",
            "DURING_YOUR_TURN", "DURING_OPPONENT_TURN",
            "FIRST_ATTACK", "EVENT_FILTER_MATCH",
            "OPTIONAL_EFFECT_EXECUTED", "INPUT_VALUE_MATCH"
        ]
        self.populate_combo(self.cond_type_combo, cond_types)
        self.cond_type_combo.currentIndexChanged.connect(self.on_cond_type_changed)
        self.cond_type_combo.currentIndexChanged.connect(self.dataChanged.emit)

        layout.addWidget(QLabel(tr("Type")), 0, 0)
        layout.addWidget(self.cond_type_combo, 0, 1)

        # Value Row
        self.lbl_val = QLabel(tr("Value"))
        self.cond_val_spin = QSpinBox()
        self.cond_val_spin.setRange(-9999, 9999)
        self.cond_val_spin.valueChanged.connect(self.dataChanged.emit)

        layout.addWidget(self.lbl_val, 1, 0)
        layout.addWidget(self.cond_val_spin, 1, 1)

        # String Row
        self.lbl_str = QLabel(tr("String Value"))
        self.cond_str_edit = QLineEdit()
        self.cond_str_edit.textChanged.connect(self.dataChanged.emit)

        layout.addWidget(self.lbl_str, 2, 0)
        layout.addWidget(self.cond_str_edit, 2, 1)

        # Filter Widget
        self.cond_filter = FilterEditorWidget()
        self.cond_filter.filterChanged.connect(self.dataChanged.emit)
        self.cond_filter.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})
        self.cond_filter.setVisible(False)
        layout.addWidget(self.cond_filter, 3, 0, 1, 2)

        # Initial Update
        self.update_ui_visibility("NONE")

    def populate_combo(self, combo, items):
        combo.clear()
        for item in items:
            combo.addItem(tr(item), item)

    def on_cond_type_changed(self):
        ctype = self.cond_type_combo.currentData()
        self.update_ui_visibility(ctype)
        # dataChanged is connected directly

    def update_ui_visibility(self, condition_type):
        config = cast(dict[str, Any], CONDITION_UI_CONFIG.get(condition_type, CONDITION_UI_CONFIG["NONE"]))

        show_val = config.get("show_val", True)
        label_val = config.get("label_val", "Value")

        show_str = config.get("show_str", True)
        label_str = config.get("label_str", "String Value")

        show_filter = config.get("show_filter", False)

        self.lbl_val.setText(tr(label_val))
        self.lbl_val.setVisible(show_val)
        self.cond_val_spin.setVisible(show_val)

        self.lbl_str.setText(tr(label_str))
        self.lbl_str.setVisible(show_str)
        self.cond_str_edit.setVisible(show_str)

        self.cond_filter.setVisible(show_filter)

    def set_data(self, data):
        self.blockSignals(True)

        ctype = data.get('type', 'NONE')
        idx = self.cond_type_combo.findData(ctype)
        if idx >= 0:
            self.cond_type_combo.setCurrentIndex(idx)
        else:
            self.cond_type_combo.setCurrentIndex(0)

        self.cond_val_spin.setValue(data.get('value', 0))
        self.cond_str_edit.setText(data.get('str_val', ''))
        self.cond_filter.set_data(data.get('filter', {}))

        self.update_ui_visibility(ctype)

        self.blockSignals(False)

    def get_data(self):
        ctype = self.cond_type_combo.currentData()
        data = {
            "type": ctype,
            "value": self.cond_val_spin.value()
        }

        str_val = self.cond_str_edit.text()
        if str_val:
            data['str_val'] = str_val

        if self.cond_filter.isVisible():
            data['filter'] = self.cond_filter.get_data()

        return data

    def blockSignals(self, block):
        super().blockSignals(block)
        self.cond_type_combo.blockSignals(block)
        self.cond_val_spin.blockSignals(block)
        self.cond_str_edit.blockSignals(block)
        self.cond_filter.blockSignals(block)
