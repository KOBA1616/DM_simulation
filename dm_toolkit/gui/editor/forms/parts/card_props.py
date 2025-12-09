from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QLabel, QGridLayout, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector

class CardPropertiesWidget(QWidget):
    """
    Reusable widget for editing card properties (Name, Cost, Power, etc.)
    Can be used for the main card or the spell side of a Twinpact card.
    """

    # Signal to notify parent of changes
    dataChanged = pyqtSignal()

    def __init__(self, parent=None, is_spell_side=False):
        super().__init__(parent)
        self.is_spell_side = is_spell_side
        self.keyword_checks = {}
        self._is_populating = False
        self.setup_ui()

    def setup_ui(self):
        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Name
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.emit_change)
        self.layout.addRow(tr("Name"), self.name_edit)

        # Civilization
        self.civ_selector = CivilizationSelector()
        self.civ_selector.changed.connect(self.emit_change)
        self.layout.addRow(tr("Civilization"), self.civ_selector)

        # Type (Fixed to SPELL if spell side, though technically modifiable)
        self.type_combo = QComboBox()
        types = ["CREATURE", "SPELL", "EVOLUTION_CREATURE", "CROSS_GEAR", "WEAPON"] # Extended types if needed
        if self.is_spell_side:
            # Usually Spell Side is just SPELL, but let's allow it just in case
            pass
        self.populate_combo(self.type_combo, types)
        self.type_combo.currentIndexChanged.connect(self.emit_change)
        self.layout.addRow(tr("Type"), self.type_combo)

        # Cost
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.cost_spin.valueChanged.connect(self.emit_change)
        self.layout.addRow(tr("Cost"), self.cost_spin)

        # Power (Only for creatures usually)
        self.power_spin = QSpinBox()
        self.power_spin.setRange(0, 99999)
        self.power_spin.setSingleStep(500)
        self.power_spin.valueChanged.connect(self.emit_change)
        self.layout.addRow(tr("Power"), self.power_spin)

        # Races
        self.races_edit = QLineEdit()
        self.races_edit.textChanged.connect(self.emit_change)
        self.layout.addRow(tr("Races"), self.races_edit)

        # Keywords Section
        self.kw_group = QGroupBox(tr("Keywords"))
        kw_layout = QGridLayout(self.kw_group)

        keywords_list = [
            "speed_attacker", "blocker", "slayer",
            "double_breaker", "triple_breaker", "shield_trigger",
            "evolution", "just_diver", "mach_fighter", "g_strike",
            "hyper_energy", "shield_burn"
        ]

        kw_map = {
            "speed_attacker": "Speed Attacker",
            "blocker": "Blocker",
            "slayer": "Slayer",
            "double_breaker": "Double Breaker",
            "triple_breaker": "Triple Breaker",
            "shield_trigger": "Shield Trigger",
            "evolution": "Evolution",
            "just_diver": "Just Diver",
            "mach_fighter": "Mach Fighter",
            "g_strike": "G Strike",
            "hyper_energy": "Hyper Energy",
            "shield_burn": "Shield Burn (Incineration)"
        }

        row = 0
        col = 0
        for k in keywords_list:
            cb = QCheckBox(tr(kw_map[k]))
            kw_layout.addWidget(cb, row, col)
            self.keyword_checks[k] = cb
            cb.stateChanged.connect(self.emit_change)

            col += 1
            if col > 2: # 3 columns
                col = 0
                row += 1

        self.layout.addRow(self.kw_group)

        if self.is_spell_side:
            # Configure UI for spell side defaults
            # Hide Power? Spells don't have power usually.
            # But kept visible just in case of weird rules.
            # Maybe disable them by default?
            pass

    def populate_combo(self, combo: QComboBox, items: list):
        combo.clear()
        for item in items:
            combo.addItem(str(item), item)

    def load_data(self, data):
        self._is_populating = True
        try:
            self.name_edit.setText(data.get('name', ''))

            civs = data.get('civilizations')
            if not civs:
                civ_single = data.get('civilization')
                if civ_single:
                    civs = [civ_single]
            self.civ_selector.set_selected_civs(civs)

            type_val = data.get('type', 'SPELL' if self.is_spell_side else 'CREATURE')
            idx = self.type_combo.findData(type_val)
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)

            self.cost_spin.setValue(int(data.get('cost', 0)))
            self.power_spin.setValue(int(data.get('power', 0)))

            races = data.get('races', [])
            if isinstance(races, list):
                self.races_edit.setText(", ".join(races))
            else:
                self.races_edit.setText(str(races))

            kw_data = data.get('keywords', {})
            for k, cb in self.keyword_checks.items():
                cb.setChecked(kw_data.get(k, False))

        finally:
            self._is_populating = False

    def get_data(self):
        data = {}
        data['name'] = self.name_edit.text()
        data['civilizations'] = self.civ_selector.get_selected_civs()
        data['type'] = self.type_combo.currentData()
        data['cost'] = self.cost_spin.value()
        data['power'] = self.power_spin.value()

        races_str = self.races_edit.text()
        data['races'] = [r.strip() for r in races_str.split(',') if r.strip()]

        kw_data = {}
        for k, cb in self.keyword_checks.items():
            if cb.isChecked():
                kw_data[k] = True
        data['keywords'] = kw_data

        return data

    def emit_change(self):
        if not self._is_populating:
            self.dataChanged.emit()

    def block_signals_all(self, block):
        self.name_edit.blockSignals(block)
        self.civ_selector.blockSignals(block)
        self.type_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.power_spin.blockSignals(block)
        self.races_edit.blockSignals(block)
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
