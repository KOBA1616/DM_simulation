# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QLabel
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from typing import Any, Dict
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.editor.forms.parts.reaction_condition_widget import ReactionConditionWidget

class ReactionEditForm(BaseEditForm):
    """
    Form to edit a single ReactionAbility item.
    Replaces the list-based ReactionWidget for detail editing.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Safe defaults for headless/static contexts
        self.type_combo = getattr(self, 'type_combo', None)
        self.cost_spin = getattr(self, 'cost_spin', None)
        self.zone_edit = getattr(self, 'zone_edit', None)
        self.reaction_condition = getattr(self, 'reaction_condition', None)

        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        # Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["NONE", "NINJA_STRIKE", "STRIKE_BACK", "REVOLUTION_0_TRIGGER"])
        self.type_combo.currentIndexChanged.connect(self.update_data)
        self.type_combo.currentIndexChanged.connect(self.update_visibility)
        layout.addRow(tr("Type"), self.type_combo)

        # Cost
        self.label_cost = QLabel(tr("Cost / Requirement"))
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.cost_spin.valueChanged.connect(self.update_data)
        layout.addRow(self.label_cost, self.cost_spin)

        # Zone
        self.label_zone = QLabel(tr("Zone"))
        self.zone_edit = QComboBox()
        self.zone_edit.addItems(["HAND", "GRAVEYARD", "MANA_ZONE"])
        self.zone_edit.currentTextChanged.connect(self.update_data)
        layout.addRow(self.label_zone, self.zone_edit)

        # Condition Widget (Extracted)
        self.reaction_condition = ReactionConditionWidget()
        self.reaction_condition.dataChanged.connect(self.update_data)
        layout.addRow(self.reaction_condition)

        # Initial visibility
        self.update_visibility()

    def update_visibility(self):
        rtype = self.type_combo.currentText()

        # Defaults
        self.label_cost.setVisible(True)
        self.cost_spin.setVisible(True)

        # Propagate visibility update to condition widget
        self.reaction_condition.update_visibility(rtype)

        if rtype == "STRIKE_BACK":
            self.label_cost.setVisible(False)
            self.cost_spin.setVisible(False)
        elif rtype == "REVOLUTION_0_TRIGGER":
            self.label_cost.setVisible(False)
            self.cost_spin.setVisible(False)

    def _load_ui_from_data(self, data, item):
        # data is passed directly from BaseEditForm.load_data
        if data is None:
             data = {}

        self.set_combo_text(self.type_combo, data.get('type', 'NONE'))
        self.cost_spin.setValue(data.get('cost', 0))
        self.set_combo_text(self.zone_edit, data.get('zone', 'HAND'))

        cond = data.get('condition', {})
        self.reaction_condition.set_data(cond)

        self.update_visibility()

    def _save_ui_to_data(self, data):
        data['type'] = self.type_combo.currentText()
        data['cost'] = self.cost_spin.value()
        data['zone'] = self.zone_edit.currentText()
        data['condition'] = self.reaction_condition.get_data()

    def _get_display_text(self, data):
        return f"{tr('Reaction Ability')}: {data.get('type', 'NONE')}"

    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.zone_edit.blockSignals(block)
        self.reaction_condition.blockSignals(block)

    def set_combo_text(self, combo, text):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)
