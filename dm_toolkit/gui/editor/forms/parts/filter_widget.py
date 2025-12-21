# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QGridLayout, QLabel, QCheckBox, QComboBox, QSpinBox,
    QLineEdit, QVBoxLayout
)
from PyQt6.QtCore import pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector
from dm_toolkit.consts import ZONES, CARD_TYPES

class FilterEditorWidget(QWidget):
    """
    Reusable widget for editing FilterDef properties.
    Handles Zones, Types, Civs, Races, Costs, Powers, Flags, and Count mode.
    """

    # Signal emitted when any filter property changes
    filterChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # Using a vertical layout to stack groups
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Basic Properties (Zones, Types, Civs)
        self.basic_group = QGroupBox(tr("Basic Filter"))
        basic_layout = QGridLayout(self.basic_group)
        main_layout.addWidget(self.basic_group)

        # Zones
        self.zone_label = QLabel(tr("Zones:"))
        basic_layout.addWidget(self.zone_label, 0, 0)
        self.zone_checks = {}
        # Use centralized ZONES definition
        zones = ZONES
        zone_grid = QGridLayout()
        for i, z in enumerate(zones):
            cb = QCheckBox(tr(z))
            cb.setToolTip(tr(f"Include {z} in target selection"))
            self.zone_checks[z] = cb
            zone_grid.addWidget(cb, i//2, i%2)
            cb.stateChanged.connect(self.filterChanged.emit)
        basic_layout.addLayout(zone_grid, 0, 1)

        # Types
        self.type_label = QLabel(tr("Types:"))
        basic_layout.addWidget(self.type_label, 1, 0)
        self.type_checks = {}
        # Use centralized CARD_TYPES definition
        types = CARD_TYPES
        type_grid = QGridLayout()
        for i, t in enumerate(types):
            cb = QCheckBox(tr(t))
            self.type_checks[t] = cb
            type_grid.addWidget(cb, i//3, i%3) # Adjusted grid to fit more types
            cb.stateChanged.connect(self.filterChanged.emit)
        basic_layout.addLayout(type_grid, 1, 1)

        # Civilizations
        self.civ_label = QLabel(tr("Civilizations:"))
        basic_layout.addWidget(self.civ_label, 2, 0)
        self.civ_selector = CivilizationSelector()
        self.civ_selector.changed.connect(self.filterChanged.emit)
        basic_layout.addWidget(self.civ_selector, 2, 1)

        # Races
        self.race_label = QLabel(tr("Races:"))
        basic_layout.addWidget(self.race_label, 3, 0)
        self.races_edit = QLineEdit()
        self.races_edit.setPlaceholderText(tr("Comma separated races (e.g. Dragon, Cyber Lord)"))
        self.races_edit.textChanged.connect(self.filterChanged.emit)
        basic_layout.addWidget(self.races_edit, 3, 1)

        # 2. Stats (Cost, Power)
        self.stats_group = QGroupBox(tr("Stats Filter"))
        stats_layout = QGridLayout(self.stats_group)
        main_layout.addWidget(self.stats_group)

        stats_layout.addWidget(QLabel(tr("Cost:")), 0, 0)
        self.min_cost_spin = QSpinBox()
        self.min_cost_spin.setRange(-1, 99) # -1 means None
        self.min_cost_spin.setValue(-1)
        self.min_cost_spin.setSpecialValueText(tr("Any"))

        self.max_cost_spin = QSpinBox()
        self.max_cost_spin.setRange(-1, 99)
        self.max_cost_spin.setValue(-1)
        self.max_cost_spin.setSpecialValueText(tr("Any"))

        cost_layout = QGridLayout()
        cost_layout.addWidget(QLabel("Min:"), 0, 0)
        cost_layout.addWidget(self.min_cost_spin, 0, 1)
        cost_layout.addWidget(QLabel("Max:"), 0, 2)
        cost_layout.addWidget(self.max_cost_spin, 0, 3)
        stats_layout.addLayout(cost_layout, 0, 1)

        self.min_cost_spin.valueChanged.connect(self.filterChanged.emit)
        self.max_cost_spin.valueChanged.connect(self.filterChanged.emit)

        stats_layout.addWidget(QLabel(tr("Power:")), 1, 0)
        self.min_power_spin = QSpinBox()
        self.min_power_spin.setRange(-1, 99999)
        self.min_power_spin.setSingleStep(500)
        self.min_power_spin.setValue(-1)
        self.min_power_spin.setSpecialValueText(tr("Any"))

        self.max_power_spin = QSpinBox()
        self.max_power_spin.setRange(-1, 99999)
        self.max_power_spin.setSingleStep(500)
        self.max_power_spin.setValue(-1)
        self.max_power_spin.setSpecialValueText(tr("Any"))

        power_layout = QGridLayout()
        power_layout.addWidget(QLabel("Min:"), 0, 0)
        power_layout.addWidget(self.min_power_spin, 0, 1)
        power_layout.addWidget(QLabel("Max:"), 0, 2)
        power_layout.addWidget(self.max_power_spin, 0, 3)
        stats_layout.addLayout(power_layout, 1, 1)

        self.min_power_spin.valueChanged.connect(self.filterChanged.emit)
        self.max_power_spin.valueChanged.connect(self.filterChanged.emit)

        # 3. Flags (Tapped, Blocker, Evolution)
        self.flags_group = QGroupBox(tr("Flags Filter"))
        flags_layout = QGridLayout(self.flags_group)
        main_layout.addWidget(self.flags_group)

        # Helper to create tri-state combos
        def create_tristate(label):
            l = QLabel(tr(label))
            c = QComboBox()
            c.addItem(tr("Ignore"), -1)
            c.addItem(tr("Yes (True)"), 1)
            c.addItem(tr("No (False)"), 0)
            c.currentIndexChanged.connect(self.filterChanged.emit)
            return l, c

        lbl_tapped, self.tapped_combo = create_tristate("Is Tapped?")
        flags_layout.addWidget(lbl_tapped, 0, 0)
        flags_layout.addWidget(self.tapped_combo, 0, 1)

        lbl_blocker, self.blocker_combo = create_tristate("Is Blocker?")
        flags_layout.addWidget(lbl_blocker, 1, 0)
        flags_layout.addWidget(self.blocker_combo, 1, 1)

        lbl_evo, self.evolution_combo = create_tristate("Is Evolution?")
        flags_layout.addWidget(lbl_evo, 2, 0)
        flags_layout.addWidget(self.evolution_combo, 2, 1)

        lbl_card, self.card_designation_combo = create_tristate("Card Designation")
        flags_layout.addWidget(lbl_card, 3, 0)
        flags_layout.addWidget(self.card_designation_combo, 3, 1)

        # 4. Count / Selection Mode (Keep at bottom)
        self.sel_group = QGroupBox(tr("Selection"))
        sel_layout = QGridLayout(self.sel_group)
        main_layout.addWidget(self.sel_group)

        self.mode_label = QLabel(tr("Selection Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("Selection_All"), 0)
        self.mode_combo.addItem(tr("Selection_Any"), 2)
        self.mode_combo.addItem(tr("Fixed Number"), 1)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 99)
        self.count_spin.setToolTip(tr("Number of cards to select/count."))
        self.count_spin.setVisible(False) # Default hidden

        sel_layout.addWidget(self.mode_label, 0, 0)
        sel_layout.addWidget(self.mode_combo, 0, 1)
        sel_layout.addWidget(self.count_spin, 1, 1)

        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.count_spin.valueChanged.connect(self.filterChanged.emit)

        # External control label (initially hidden)
        self.external_count_label = QLabel(tr("Defined by Input Variable"))
        self.external_count_label.setStyleSheet("color: gray; font-style: italic;")
        self.external_count_label.setVisible(False)
        sel_layout.addWidget(self.external_count_label, 0, 1)

        self.on_mode_changed() # Init visibility

    def on_mode_changed(self):
        mode = self.mode_combo.currentData()
        is_fixed = (mode == 1)
        self.count_spin.setVisible(is_fixed and self.mode_combo.isVisible())
        self.filterChanged.emit()

    def set_external_count_control(self, active: bool):
        """
        If active, hides manual count controls and shows a label indicating
        that the count is determined by an external variable (input_value_key).
        """
        self.mode_combo.setVisible(not active)
        self.count_spin.setVisible(False) # Always hide first, then re-eval
        self.external_count_label.setVisible(active)

        if not active:
             self.on_mode_changed() # Restore state

    def set_data(self, filt_data):
        """
        Populate UI from dictionary (FilterDef).
        """
        self.blockSignals(True)
        if not filt_data: filt_data = {}

        # Zones
        zones = filt_data.get('zones', [])
        for z, cb in self.zone_checks.items():
            cb.setChecked(z in zones)

        # Types
        types = filt_data.get('types', [])
        for t, cb in self.type_checks.items():
            cb.setChecked(t in types)

        # Civs
        civs = filt_data.get('civilizations', [])
        self.civ_selector.set_selected_civs(civs)

        # Races
        races = filt_data.get('races', [])
        self.races_edit.setText(", ".join(races))

        # Costs
        self.min_cost_spin.setValue(filt_data.get('min_cost', -1) if filt_data.get('min_cost') is not None else -1)
        self.max_cost_spin.setValue(filt_data.get('max_cost', -1) if filt_data.get('max_cost') is not None else -1)

        # Powers
        self.min_power_spin.setValue(filt_data.get('min_power', -1) if filt_data.get('min_power') is not None else -1)
        self.max_power_spin.setValue(filt_data.get('max_power', -1) if filt_data.get('max_power') is not None else -1)

        # Flags
        def set_tristate(combo, val):
            if val is None: combo.setCurrentIndex(0) # Ignore
            elif val is True: combo.setCurrentIndex(1) # Yes
            else: combo.setCurrentIndex(2) # No

        set_tristate(self.tapped_combo, filt_data.get('is_tapped'))
        set_tristate(self.blocker_combo, filt_data.get('is_blocker'))
        set_tristate(self.evolution_combo, filt_data.get('is_evolution'))
        set_tristate(self.card_designation_combo, filt_data.get('is_card_designation'))

        # Count
        count_val = filt_data.get('count', 0)
        if count_val == 1:
            self.mode_combo.setCurrentIndex(1) # Any
            self.count_spin.setValue(1)
            self.count_spin.setVisible(False)
        elif count_val > 1:
            self.mode_combo.setCurrentIndex(2) # Fixed
            self.count_spin.setValue(count_val)
            self.count_spin.setVisible(True)
        else:
            self.mode_combo.setCurrentIndex(0) # All
            self.count_spin.setValue(1)
            self.count_spin.setVisible(False)

        self.blockSignals(False)

    def get_data(self):
        """
        Return dictionary (FilterDef) from UI.
        """
        filt = {}

        zones = [z for z, cb in self.zone_checks.items() if cb.isChecked()]
        if zones: filt['zones'] = zones

        types = [t for t, cb in self.type_checks.items() if cb.isChecked()]
        if types: filt['types'] = types

        civs = self.civ_selector.get_selected_civs()
        if civs: filt['civilizations'] = civs

        races_str = self.races_edit.text()
        races = [r.strip() for r in races_str.split(',') if r.strip()]
        if races: filt['races'] = races

        if self.min_cost_spin.value() != -1: filt['min_cost'] = self.min_cost_spin.value()
        if self.max_cost_spin.value() != -1: filt['max_cost'] = self.max_cost_spin.value()

        if self.min_power_spin.value() != -1: filt['min_power'] = self.min_power_spin.value()
        if self.max_power_spin.value() != -1: filt['max_power'] = self.max_power_spin.value()

        def get_tristate(combo):
            idx = combo.currentIndex()
            if idx == 0: return None
            return (idx == 1)

        val_tapped = get_tristate(self.tapped_combo)
        if val_tapped is not None: filt['is_tapped'] = val_tapped

        val_blocker = get_tristate(self.blocker_combo)
        if val_blocker is not None: filt['is_blocker'] = val_blocker

        val_evo = get_tristate(self.evolution_combo)
        if val_evo is not None: filt['is_evolution'] = val_evo

        val_card = get_tristate(self.card_designation_combo)
        if val_card is not None: filt['is_card_designation'] = val_card

        mode = self.mode_combo.currentData()
        if mode == 1:
            count = self.count_spin.value()
            if count > 0: filt['count'] = count
        elif mode == 2:
            filt['count'] = 1

        return filt

    def blockSignals(self, block):
        super().blockSignals(block)
        for cb in self.zone_checks.values(): cb.blockSignals(block)
        for cb in self.type_checks.values(): cb.blockSignals(block)
        self.civ_selector.blockSignals(block)
        self.races_edit.blockSignals(block)
        self.min_cost_spin.blockSignals(block)
        self.max_cost_spin.blockSignals(block)
        self.min_power_spin.blockSignals(block)
        self.max_power_spin.blockSignals(block)
        self.tapped_combo.blockSignals(block)
        self.blocker_combo.blockSignals(block)
        self.evolution_combo.blockSignals(block)
        self.card_designation_combo.blockSignals(block)
        self.mode_combo.blockSignals(block)
        self.count_spin.blockSignals(block)

    def set_visible_sections(self, sections: dict):
        """
        Toggle visibility of filter groups.
        sections: dict mapping section name ('basic', 'stats', 'flags', 'selection') to bool.
        """
        if 'basic' in sections: self.basic_group.setVisible(sections['basic'])
        if 'stats' in sections: self.stats_group.setVisible(sections['stats'])
        if 'flags' in sections: self.flags_group.setVisible(sections['flags'])
        if 'selection' in sections: self.sel_group.setVisible(sections['selection'])

    def set_allowed_fields(self, allowed_fields: list):
        """
        Toggle visibility of specific fields inside Basic group.
        allowed_fields: list of field keys ('zones', 'types', 'civilizations', 'races').
        If list is None, all are shown.
        """
        if allowed_fields is None:
            # Show all
            self.zone_label.setVisible(True)
            for cb in self.zone_checks.values(): cb.setVisible(True)
            self.type_label.setVisible(True)
            for cb in self.type_checks.values(): cb.setVisible(True)
            self.civ_label.setVisible(True)
            self.civ_selector.setVisible(True)
            self.race_label.setVisible(True)
            self.races_edit.setVisible(True)
            return

        is_allowed = lambda f: f in allowed_fields

        self.zone_label.setVisible(is_allowed('zones'))
        for cb in self.zone_checks.values():
            cb.setVisible(is_allowed('zones'))

        self.type_label.setVisible(is_allowed('types'))
        for cb in self.type_checks.values():
            cb.setVisible(is_allowed('types'))

        self.civ_label.setVisible(is_allowed('civilizations'))
        self.civ_selector.setVisible(is_allowed('civilizations'))

        self.race_label.setVisible(is_allowed('races'))
        self.races_edit.setVisible(is_allowed('races'))
