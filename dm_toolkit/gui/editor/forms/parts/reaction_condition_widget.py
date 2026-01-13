# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QComboBox, QCheckBox, QSpinBox, QLabel
)
from PyQt6.QtCore import pyqtSignal
from dm_toolkit.gui.i18n import tr

class ReactionConditionWidget(QGroupBox):
    """
    Widget to edit ReactionCondition fields.
    Extracts condition logic from ReactionEditForm for standardization.
    """
    dataChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(tr("Condition"), parent)
        # Safe defaults
        self.trigger_event_combo = getattr(self, 'trigger_event_combo', None)
        self.civ_match_check = getattr(self, 'civ_match_check', None)
        self.shield_civ_match_check = getattr(self, 'shield_civ_match_check', None)
        self.mana_min_spin = getattr(self, 'mana_min_spin', None)

        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.label_trigger = QLabel(tr("Trigger Event"))
        self.trigger_event_combo = QComboBox()
        self.trigger_event_combo.addItems(["NONE", "ON_BLOCK_OR_ATTACK", "ON_SHIELD_ADD", "ON_ATTACK_PLAYER"])
        self.trigger_event_combo.currentTextChanged.connect(self.dataChanged)
        layout.addRow(self.label_trigger, self.trigger_event_combo)

        self.civ_match_check = QCheckBox(tr("Civilization Match Required"))
        self.civ_match_check.stateChanged.connect(self.dataChanged)
        layout.addRow(self.civ_match_check)

        self.shield_civ_match_check = QCheckBox(tr("Same Civilization Shield Required"))
        self.shield_civ_match_check.stateChanged.connect(self.dataChanged)
        layout.addRow(self.shield_civ_match_check)

        self.label_mana = QLabel(tr("Min Mana Required"))
        self.mana_min_spin = QSpinBox()
        self.mana_min_spin.setRange(0, 99)
        self.mana_min_spin.valueChanged.connect(self.dataChanged)
        layout.addRow(self.label_mana, self.mana_min_spin)

    def set_data(self, cond_data):
        self.blockSignals(True)
        self.trigger_event_combo.setCurrentText(cond_data.get('trigger_event', 'NONE'))
        self.civ_match_check.setChecked(cond_data.get('civilization_match', False))
        self.shield_civ_match_check.setChecked(cond_data.get('same_civilization_shield', False))
        self.mana_min_spin.setValue(cond_data.get('mana_count_min', 0))
        self.blockSignals(False)

    def get_data(self):
        return {
            'trigger_event': self.trigger_event_combo.currentText(),
            'civilization_match': self.civ_match_check.isChecked(),
            'same_civilization_shield': self.shield_civ_match_check.isChecked(),
            'mana_count_min': self.mana_min_spin.value()
        }

    def update_visibility(self, rtype):
        """Updates visibility of condition fields based on the reaction type."""
        # Reset defaults
        self.label_trigger.setVisible(True)
        self.trigger_event_combo.setVisible(True)
        self.civ_match_check.setVisible(True)
        self.shield_civ_match_check.setVisible(True)
        self.label_mana.setVisible(True)
        self.mana_min_spin.setVisible(True)

        if rtype == "STRIKE_BACK":
            # Strike Back: Needs Shield match. Usually no Mana min.
            self.label_mana.setVisible(False)
            self.mana_min_spin.setVisible(False)
            self.civ_match_check.setVisible(False) # Usually implied by shield match or not present
            self.shield_civ_match_check.setVisible(True)

        elif rtype == "NINJA_STRIKE":
            # Ninja Strike: Needs Mana min (cost is separate). Civ match usually required.
            self.shield_civ_match_check.setVisible(False)

        elif rtype == "REVOLUTION_0_TRIGGER":
            # Revolution 0: Special condition.
            self.shield_civ_match_check.setVisible(False)
            # Often checks shield count = 0, which is implicit in the type or handled by extra conditions?
            # For now standard fields.
