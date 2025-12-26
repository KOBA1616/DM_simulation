# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QCheckBox, QLabel, QGroupBox
)
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
from typing import Any, Dict
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm

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
        self.cond_group = getattr(self, 'cond_group', None)
        self.trigger_event_combo = getattr(self, 'trigger_event_combo', None)
        try:
            self.setup_ui()
        except Exception:
            pass

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

        # Condition Group
        self.cond_group = QGroupBox(tr("Condition"))
        cond_layout = QFormLayout(self.cond_group)

        self.label_trigger = QLabel(tr("Trigger Event"))
        self.trigger_event_combo = QComboBox()
        self.trigger_event_combo.addItems(["NONE", "ON_BLOCK_OR_ATTACK", "ON_SHIELD_ADD", "ON_ATTACK_PLAYER"])
        self.trigger_event_combo.currentTextChanged.connect(self.update_data)
        cond_layout.addRow(self.label_trigger, self.trigger_event_combo)

        self.civ_match_check = QCheckBox(tr("Civilization Match Required"))
        self.civ_match_check.stateChanged.connect(self.update_data)
        cond_layout.addRow(self.civ_match_check)

        self.shield_civ_match_check = QCheckBox(tr("Same Civilization Shield Required"))
        self.shield_civ_match_check.stateChanged.connect(self.update_data)
        cond_layout.addRow(self.shield_civ_match_check)

        self.label_mana = QLabel(tr("Min Mana Required"))
        self.mana_min_spin = QSpinBox()
        self.mana_min_spin.setRange(0, 99)
        self.mana_min_spin.valueChanged.connect(self.update_data)
        cond_layout.addRow(self.label_mana, self.mana_min_spin)

        layout.addRow(self.cond_group)

        # Initial visibility
        self.update_visibility()

    def update_visibility(self):
        rtype = self.type_combo.currentText()

        # Defaults
        self.label_cost.setVisible(True)
        self.cost_spin.setVisible(True)
        self.label_mana.setVisible(True)
        self.mana_min_spin.setVisible(True)
        self.civ_match_check.setVisible(True)
        self.shield_civ_match_check.setVisible(True)

        if rtype == "STRIKE_BACK":
            self.label_cost.setVisible(False)
            self.cost_spin.setVisible(False)
            self.label_mana.setVisible(False)
            self.mana_min_spin.setVisible(False)
            self.civ_match_check.setVisible(False)
            self.shield_civ_match_check.setVisible(True)
        elif rtype == "NINJA_STRIKE":
            self.shield_civ_match_check.setVisible(False)
        elif rtype == "REVOLUTION_0_TRIGGER":
            self.label_cost.setVisible(False)
            self.cost_spin.setVisible(False)
            self.shield_civ_match_check.setVisible(False)

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.set_combo_text(self.type_combo, data.get('type', 'NONE'))
        self.cost_spin.setValue(data.get('cost', 0))
        self.set_combo_text(self.zone_edit, data.get('zone', 'HAND'))

        cond = data.get('condition', {})
        self.set_combo_text(self.trigger_event_combo, cond.get('trigger_event', 'NONE'))
        self.civ_match_check.setChecked(cond.get('civilization_match', False))
        self.shield_civ_match_check.setChecked(cond.get('same_civilization_shield', False))
        self.mana_min_spin.setValue(cond.get('mana_count_min', 0))

        self.update_visibility()

    def _save_data(self, data):
        data['type'] = self.type_combo.currentText()
        data['cost'] = self.cost_spin.value()
        data['zone'] = self.zone_edit.currentText()

        cond: Dict[str, Any] = {}
        cond['trigger_event'] = self.trigger_event_combo.currentText()
        cond['civilization_match'] = bool(self.civ_match_check.isChecked())
        cond['same_civilization_shield'] = bool(self.shield_civ_match_check.isChecked())
        cond['mana_count_min'] = int(self.mana_min_spin.value())
        data['condition'] = cond

    def _get_display_text(self, data):
        return f"{tr('Reaction Ability')}: {data.get('type', 'NONE')}"

    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.zone_edit.blockSignals(block)
        self.trigger_event_combo.blockSignals(block)
        self.civ_match_check.blockSignals(block)
        self.shield_civ_match_check.blockSignals(block)
        self.mana_min_spin.blockSignals(block)

    def set_combo_text(self, combo, text):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)
