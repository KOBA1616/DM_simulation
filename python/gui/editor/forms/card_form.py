from PyQt6.QtWidgets import QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox, QCheckBox, QLabel, QGridLayout, QGroupBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from gui.localization import tr

class CardEditForm(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.keyword_checks = {} # Map key -> QCheckBox
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 9999)
        layout.addRow(tr("ID"), self.id_spin)

        self.name_edit = QLineEdit()
        layout.addRow(tr("Name"), self.name_edit)

        self.civ_combo = QComboBox()
        civs = ["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"]
        civ_colors = {
            "LIGHT": QColor("goldenrod"),
            "WATER": QColor("blue"),
            "DARKNESS": QColor("darkGray"),
            "FIRE": QColor("red"),
            "NATURE": QColor("green"),
            "ZERO": QColor("gray")
        }
        for c in civs:
            self.civ_combo.addItem(tr(c), c)
            idx = self.civ_combo.count() - 1
            if c in civ_colors:
                self.civ_combo.setItemData(idx, civ_colors[c], Qt.ItemDataRole.ForegroundRole)
        layout.addRow(tr("Civilization"), self.civ_combo)

        self.type_combo = QComboBox()
        types = ["CREATURE", "SPELL", "EVOLUTION_CREATURE"]
        for t in types:
            self.type_combo.addItem(tr(t), t)
        layout.addRow(tr("Type"), self.type_combo)

        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        layout.addRow(tr("Cost"), self.cost_spin)

        self.power_spin = QSpinBox()
        self.power_spin.setRange(0, 99999)
        self.power_spin.setSingleStep(500)
        layout.addRow(tr("Power"), self.power_spin)

        self.races_edit = QLineEdit()
        layout.addRow(tr("Races"), self.races_edit)

        # Keywords Section
        kw_group = QGroupBox(tr("Keywords"))
        kw_layout = QGridLayout(kw_group)

        # List of keywords to support in UI
        keywords_list = [
            "speed_attacker", "blocker", "slayer",
            "double_breaker", "triple_breaker", "shield_trigger",
            "evolution", "just_diver", "mach_fighter", "g_strike",
            "hyper_energy", "shield_burn"
        ]

        # Use localized names for labels if possible
        # Map snake_case to Title Case for lookup in tr() if needed,
        # or just use tr(Title Case)
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
            cb.stateChanged.connect(self.update_data) # Connect directly

            col += 1
            if col > 2: # 3 columns
                col = 0
                row += 1

        layout.addRow(kw_group)

        # Connect signals (existing)
        self.id_spin.valueChanged.connect(self.update_data)
        self.name_edit.textChanged.connect(self.update_data)
        self.civ_combo.currentIndexChanged.connect(self.update_data)
        self.type_combo.currentIndexChanged.connect(self.update_data)
        self.cost_spin.valueChanged.connect(self.update_data)
        self.power_spin.valueChanged.connect(self.update_data)
        self.races_edit.textChanged.connect(self.update_data)

    def set_data(self, item):
        self.current_item = item
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.block_signals(True)
        self.id_spin.setValue(data.get('id', 0))
        self.name_edit.setText(data.get('name', ''))

        civ_idx = self.civ_combo.findData(data.get('civilization', 'FIRE'))
        if civ_idx >= 0:
            self.civ_combo.setCurrentIndex(civ_idx)

        type_idx = self.type_combo.findData(data.get('type', 'CREATURE'))
        if type_idx >= 0:
            self.type_combo.setCurrentIndex(type_idx)

        self.cost_spin.setValue(data.get('cost', 0))
        self.power_spin.setValue(data.get('power', 0))
        self.races_edit.setText(", ".join(data.get('races', [])))

        # Load Keywords
        # Keywords might be a dict { "speed_attacker": true } in JSON
        kw_data = data.get('keywords', {})
        for k, cb in self.keyword_checks.items():
            is_checked = kw_data.get(k, False)
            cb.setChecked(is_checked)

        self.block_signals(False)

    def update_data(self):
        if not self.current_item: return
        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)

        data['id'] = self.id_spin.value()
        data['name'] = self.name_edit.text()
        data['civilization'] = self.civ_combo.currentData()
        data['type'] = self.type_combo.currentData()
        data['cost'] = self.cost_spin.value()
        data['power'] = self.power_spin.value()
        races_str = self.races_edit.text()
        data['races'] = [r.strip() for r in races_str.split(',') if r.strip()]

        # Update Keywords
        kw_data = {}
        for k, cb in self.keyword_checks.items():
            if cb.isChecked():
                kw_data[k] = True

        # Only set if not empty to keep JSON clean? Or explicit?
        # Explicit is better for now.
        data['keywords'] = kw_data

        self.current_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        self.current_item.setText(f"{data['id']} - {data['name']}")

    def block_signals(self, block):
        self.id_spin.blockSignals(block)
        self.name_edit.blockSignals(block)
        self.civ_combo.blockSignals(block)
        self.type_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.power_spin.blockSignals(block)
        self.races_edit.blockSignals(block)
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
