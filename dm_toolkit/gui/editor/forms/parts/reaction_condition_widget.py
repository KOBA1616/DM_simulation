# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QComboBox, QCheckBox, QSpinBox, QLabel
)
from PyQt6.QtCore import pyqtSignal
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect

# Mapping for known reaction types to visibility overrides (module-level for testability)
REACTION_VISIBILITY_MAP = {
    'STRIKE_BACK': {
        'label_mana': False,
        'mana_min_spin': False,
        'civ_match_check': False,
        'shield_civ_match_check': True,
    },
    'NINJA_STRIKE': {
        'shield_civ_match_check': False,
    },
    'REVOLUTION_0_TRIGGER': {
        'shield_civ_match_check': False,
    },
}

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
        # 再発防止: addItem(表示テキスト, 生値) パターン。get_data() は currentData() を使用する。
        for raw in ["NONE", "ON_BLOCK_OR_ATTACK", "ON_SHIELD_ADD", "ON_ATTACK_PLAYER"]:
            self.trigger_event_combo.addItem(tr(raw), raw)
        safe_connect(self.trigger_event_combo, 'currentTextChanged', self.dataChanged)
        layout.addRow(self.label_trigger, self.trigger_event_combo)

        self.civ_match_check = QCheckBox(tr("Civilization Match Required"))
        safe_connect(self.civ_match_check, 'stateChanged', self.dataChanged)
        layout.addRow(self.civ_match_check)

        self.shield_civ_match_check = QCheckBox(tr("Same Civilization Shield Required"))
        safe_connect(self.shield_civ_match_check, 'stateChanged', self.dataChanged)
        layout.addRow(self.shield_civ_match_check)

        self.label_mana = QLabel(tr("Min Mana Required"))
        self.mana_min_spin = QSpinBox()
        self.mana_min_spin.setRange(0, 99)
        safe_connect(self.mana_min_spin, 'valueChanged', self.dataChanged)
        layout.addRow(self.label_mana, self.mana_min_spin)

    def set_data(self, cond_data):
        self.blockSignals(True)
        # 再発防止: 翻訳表示コンボでは findData() で生値を検索する。
        raw = cond_data.get('trigger_event', 'NONE')
        idx = self.trigger_event_combo.findData(raw)
        if idx < 0:
            idx = self.trigger_event_combo.findText(raw)  # フォールバック
        self.trigger_event_combo.setCurrentIndex(max(0, idx))
        self.civ_match_check.setChecked(cond_data.get('civilization_match', False))
        self.shield_civ_match_check.setChecked(cond_data.get('same_civilization_shield', False))
        self.mana_min_spin.setValue(cond_data.get('mana_count_min', 0))
        self.blockSignals(False)

    def get_data(self):
        return {
            # 再発防止: 翻訳表示コンボでは currentData() で生値を取得する。
            'trigger_event': self.trigger_event_combo.currentData() or self.trigger_event_combo.currentText(),
            'civilization_match': self.civ_match_check.isChecked(),
            'same_civilization_shield': self.shield_civ_match_check.isChecked(),
            'mana_count_min': self.mana_min_spin.value()
        }

    def update_visibility(self, rtype):
        """Updates visibility of condition fields based on the reaction type."""
        # Default visibility state
        defaults = {
            'label_trigger': True,
            'trigger_event_combo': True,
            'civ_match_check': True,
            'shield_civ_match_check': True,
            'label_mana': True,
            'mana_min_spin': True,
        }

        # Compute final visibilities using module-level mapping
        vis = defaults.copy()
        vis.update(REACTION_VISIBILITY_MAP.get(rtype, {}))

        # Apply visibilities
        self.label_trigger.setVisible(vis['label_trigger'])
        self.trigger_event_combo.setVisible(vis['trigger_event_combo'])
        self.civ_match_check.setVisible(vis['civ_match_check'])
        self.shield_civ_match_check.setVisible(vis['shield_civ_match_check'])
        self.label_mana.setVisible(vis['label_mana'])
        self.mana_min_spin.setVisible(vis['mana_min_spin'])
