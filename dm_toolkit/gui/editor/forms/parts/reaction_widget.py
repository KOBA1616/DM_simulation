# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QSpinBox, QCheckBox, QLabel,
    QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr

class ReactionWidget(QWidget):
    """
    Widget to manage a list of ReactionAbility items.
    Optimized for Phase 3.1 Requirements with dynamic visibility and better context.
    """
    dataChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reactions = [] # List of ReactionAbility dicts
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: List of Reactions
        list_layout = QVBoxLayout()
        self.reaction_list = QListWidget()
        self.reaction_list.currentRowChanged.connect(self.on_selection_changed)
        list_layout.addWidget(self.reaction_list)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton(tr("Add"))
        self.btn_remove = QPushButton(tr("Remove"))
        self.btn_add.clicked.connect(self.add_reaction)
        self.btn_remove.clicked.connect(self.remove_reaction)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        list_layout.addLayout(btn_layout)

        main_layout.addLayout(list_layout, 1) # Stretch factor 1

        # Right: Editor for selected reaction
        self.editor_group = QGroupBox(tr("Reaction Details"))
        self.editor_layout = QFormLayout(self.editor_group)

        # Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["NONE", "NINJA_STRIKE", "STRIKE_BACK", "REVOLUTION_0_TRIGGER"])
        self.type_combo.currentIndexChanged.connect(self.update_current_item)
        self.type_combo.currentIndexChanged.connect(self.update_visibility)
        self.editor_layout.addRow(tr("Type"), self.type_combo)

        # Cost
        self.label_cost = QLabel(tr("Cost / Requirement"))
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.cost_spin.valueChanged.connect(self.update_current_item)
        self.editor_layout.addRow(self.label_cost, self.cost_spin)

        # Zone
        self.label_zone = QLabel(tr("Zone"))
        self.zone_edit = QComboBox()
        self.zone_edit.addItems(["HAND", "GRAVEYARD", "MANA_ZONE"]) # Usual reaction zones
        self.zone_edit.currentTextChanged.connect(self.update_current_item)
        self.editor_layout.addRow(self.label_zone, self.zone_edit)

        # Condition Details
        self.cond_group = QGroupBox(tr("Condition"))
        cond_layout = QFormLayout(self.cond_group)

        self.label_trigger = QLabel(tr("Trigger Event"))
        self.trigger_event_combo = QComboBox()
        self.trigger_event_combo.addItems(["NONE", "ON_BLOCK_OR_ATTACK", "ON_SHIELD_ADD", "ON_ATTACK_PLAYER"])
        self.trigger_event_combo.currentTextChanged.connect(self.update_current_item)
        cond_layout.addRow(self.label_trigger, self.trigger_event_combo)

        self.civ_match_check = QCheckBox(tr("Civilization Match Required"))
        self.civ_match_check.stateChanged.connect(self.update_current_item)
        cond_layout.addRow(self.civ_match_check)

        self.label_mana = QLabel(tr("Min Mana Required"))
        self.mana_min_spin = QSpinBox()
        self.mana_min_spin.setRange(0, 99)
        self.mana_min_spin.valueChanged.connect(self.update_current_item)
        cond_layout.addRow(self.label_mana, self.mana_min_spin)

        self.editor_layout.addRow(self.cond_group)

        main_layout.addWidget(self.editor_group, 2) # Stretch factor 2

        self.editor_group.setEnabled(False)

    def set_data(self, data_list):
        self.block_signals_all(True)
        self.reactions = data_list if data_list else []
        self.refresh_list()
        self.block_signals_all(False)
        self.reaction_list.setCurrentRow(-1)
        self.editor_group.setEnabled(False)

    def get_data(self):
        return self.reactions

    def refresh_list(self):
        self.reaction_list.clear()
        for r in self.reactions:
            label = f"{r.get('type', 'NONE')} ({r.get('zone', '')})"
            self.reaction_list.addItem(label)

    def add_reaction(self):
        new_reaction = {
            "type": "NINJA_STRIKE",
            "cost": 4,
            "zone": "HAND",
            "condition": {
                "trigger_event": "ON_BLOCK_OR_ATTACK",
                "civilization_match": True,
                "mana_count_min": 0
            }
        }
        self.reactions.append(new_reaction)
        self.refresh_list()
        self.reaction_list.setCurrentRow(len(self.reactions) - 1)
        self.dataChanged.emit()

    def remove_reaction(self):
        row = self.reaction_list.currentRow()
        if row >= 0:
            self.reactions.pop(row)
            self.refresh_list()
            self.dataChanged.emit()

    def on_selection_changed(self, row):
        if row < 0 or row >= len(self.reactions):
            self.editor_group.setEnabled(False)
            return

        self.editor_group.setEnabled(True)
        data = self.reactions[row]

        self.block_signals_all(True)

        self.set_combo_text(self.type_combo, data.get('type', 'NONE'))
        self.cost_spin.setValue(data.get('cost', 0))
        self.set_combo_text(self.zone_edit, data.get('zone', 'HAND'))

        cond = data.get('condition', {})
        self.set_combo_text(self.trigger_event_combo, cond.get('trigger_event', 'NONE'))
        self.civ_match_check.setChecked(cond.get('civilization_match', False))
        self.mana_min_spin.setValue(cond.get('mana_count_min', 0))

        self.update_visibility()
        self.block_signals_all(False)

    def update_visibility(self):
        rtype = self.type_combo.currentText()

        # Default visibility
        self.label_cost.setVisible(True)
        self.cost_spin.setVisible(True)
        self.label_mana.setVisible(True)
        self.mana_min_spin.setVisible(True)
        self.civ_match_check.setVisible(True)

        if rtype == "STRIKE_BACK":
            # Strike Back: No cost (0), needs Civ match.
            # Usually doesn't need Mana Min.
            self.label_cost.setVisible(False)
            self.cost_spin.setVisible(False)
            self.label_mana.setVisible(False)
            self.mana_min_spin.setVisible(False)

            # Auto-set trigger event if not set?
            # self.set_combo_text(self.trigger_event_combo, "ON_SHIELD_ADD") # Might be annoying if user wants custom

        elif rtype == "NINJA_STRIKE":
            # Ninja Strike: Needs Cost (Mana Cost), Zone (Hand).
            # Usually Civ Match is True.
            pass

        elif rtype == "REVOLUTION_0_TRIGGER":
            # Revolution 0: Needs Cost? Usually 0.
            # Trigger: ON_ATTACK_PLAYER.
            self.label_cost.setVisible(False)
            self.cost_spin.setVisible(False)

    def update_current_item(self):
        row = self.reaction_list.currentRow()
        if row < 0: return

        data = self.reactions[row]

        data['type'] = self.type_combo.currentText()
        data['cost'] = self.cost_spin.value()
        data['zone'] = self.zone_edit.currentText()

        if 'condition' not in data: data['condition'] = {}
        data['condition']['trigger_event'] = self.trigger_event_combo.currentText()
        data['condition']['civilization_match'] = self.civ_match_check.isChecked()
        data['condition']['mana_count_min'] = self.mana_min_spin.value()

        # Update list label
        item = self.reaction_list.item(row)
        if item is not None:
            item.setText(f"{data.get('type')} ({data.get('zone')})")

        self.dataChanged.emit()

    def set_combo_text(self, combo, text):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0) # Default?

    def block_signals_all(self, block):
        self.type_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.zone_edit.blockSignals(block)
        self.trigger_event_combo.blockSignals(block)
        self.civ_match_check.blockSignals(block)
        self.mana_min_spin.blockSignals(block)
