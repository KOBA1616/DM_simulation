from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QLineEdit, QVBoxLayout
)
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm

class ReactionEditForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Main Properties
        main_group = QGroupBox(tr("Reaction Properties"))
        form_layout = QFormLayout(main_group)

        # Reaction Type
        self.type_combo = QComboBox()
        self.populate_combo(self.type_combo, [
            "NONE", "NINJA_STRIKE", "STRIKE_BACK", "REVOLUTION_0_TRIGGER"
        ])
        form_layout.addRow(tr("Reaction Type"), self.type_combo)

        # Zone (Source Zone)
        self.zone_combo = QComboBox()
        self.populate_combo(self.zone_combo, ["HAND", "GRAVEYARD", "MANA_ZONE"])
        form_layout.addRow(tr("Source Zone"), self.zone_combo)

        # Cost (e.g. Ninja Strike 7)
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        form_layout.addRow(tr("Cost (e.g. NS Level)"), self.cost_spin)

        layout.addWidget(main_group)

        # Conditions
        cond_group = QGroupBox(tr("Trigger Conditions"))
        cond_layout = QFormLayout(cond_group)

        # Trigger Event
        self.trigger_event_combo = QComboBox()
        self.populate_combo(self.trigger_event_combo, [
            "NONE", "ON_BLOCK_OR_ATTACK", "ON_SHIELD_ADD"
        ])
        cond_layout.addRow(tr("Trigger Event"), self.trigger_event_combo)

        # Civilization Match (e.g. Strike Back)
        self.civ_match_check = QCheckBox(tr("Require Civilization Match"))
        cond_layout.addRow(self.civ_match_check)

        # Mana Count Min (e.g. Ninja Strike requires X mana)
        self.mana_min_spin = QSpinBox()
        self.mana_min_spin.setRange(0, 99)
        cond_layout.addRow(tr("Min Mana Required"), self.mana_min_spin)

        # Same Civ Shield (e.g. Strike Back)
        self.shield_match_check = QCheckBox(tr("Shield Matches Card Civ"))
        cond_layout.addRow(self.shield_match_check)

        layout.addWidget(cond_group)
        layout.addStretch()

        # Connect signals
        self.type_combo.currentIndexChanged.connect(self.update_data)
        self.zone_combo.currentIndexChanged.connect(self.update_data)
        self.cost_spin.valueChanged.connect(self.update_data)
        self.trigger_event_combo.currentIndexChanged.connect(self.update_data)
        self.civ_match_check.stateChanged.connect(self.update_data)
        self.mana_min_spin.valueChanged.connect(self.update_data)
        self.shield_match_check.stateChanged.connect(self.update_data)

    def _populate_ui(self, item):
        data = item.data(258) # Qt.ItemDataRole.UserRole + 2
        if not data:
            data = {}

        self.set_combo_by_data(self.type_combo, data.get('type', 'NONE'))
        self.set_combo_by_data(self.zone_combo, data.get('zone', 'HAND'))
        self.cost_spin.setValue(data.get('cost', 0))

        cond = data.get('condition', {})
        self.set_combo_by_data(self.trigger_event_combo, cond.get('trigger_event', 'NONE'))
        self.civ_match_check.setChecked(cond.get('civilization_match', False))
        self.mana_min_spin.setValue(cond.get('mana_count_min', 0))
        self.shield_match_check.setChecked(cond.get('same_civilization_shield', False))

    def _save_data(self, data):
        data['type'] = self.type_combo.currentData()
        data['zone'] = self.zone_combo.currentData()
        data['cost'] = self.cost_spin.value()

        if 'condition' not in data:
            data['condition'] = {}

        cond = data['condition']
        cond['trigger_event'] = self.trigger_event_combo.currentData()
        cond['civilization_match'] = self.civ_match_check.isChecked()
        cond['mana_count_min'] = self.mana_min_spin.value()
        cond['same_civilization_shield'] = self.shield_match_check.isChecked()

    def _get_display_text(self, data):
        rtype = data.get('type', 'NONE')
        cost = data.get('cost', 0)
        return f"{tr('Reaction')}: {tr(rtype)} ({cost})"

    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.zone_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.trigger_event_combo.blockSignals(block)
        self.civ_match_check.blockSignals(block)
        self.mana_min_spin.blockSignals(block)
        self.shield_match_check.blockSignals(block)
