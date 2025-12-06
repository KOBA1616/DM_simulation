from PyQt6.QtWidgets import QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox, QCheckBox, QLabel, QGridLayout, QGroupBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from gui.localization import tr
from gui.editor.forms.base_form import BaseEditForm

class CardEditForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.populate_combo(self.civ_combo, civs, data_func=lambda x: x)

        # Apply colors
        civ_colors = {
            "LIGHT": QColor("goldenrod"),
            "WATER": QColor("blue"),
            "DARKNESS": QColor("darkGray"),
            "FIRE": QColor("red"),
            "NATURE": QColor("green"),
            "ZERO": QColor("gray")
        }
        for i in range(self.civ_combo.count()):
            data = self.civ_combo.itemData(i)
            if data in civ_colors:
                self.civ_combo.setItemData(i, civ_colors[data], Qt.ItemDataRole.ForegroundRole)
        layout.addRow(tr("Civilization"), self.civ_combo)

        self.type_combo = QComboBox()
        types = ["CREATURE", "SPELL", "EVOLUTION_CREATURE"]
        self.populate_combo(self.type_combo, types, data_func=lambda x: x)
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
            cb.stateChanged.connect(self.update_data)

            col += 1
            if col > 2: # 3 columns
                col = 0
                row += 1

        layout.addRow(kw_group)

        # Connect signals
        self.id_spin.valueChanged.connect(self.update_data)
        self.name_edit.textChanged.connect(self.update_data)
        self.civ_combo.currentIndexChanged.connect(self.update_data)
        self.type_combo.currentIndexChanged.connect(self.update_data)
        self.cost_spin.valueChanged.connect(self.update_data)
        self.power_spin.valueChanged.connect(self.update_data)
        self.races_edit.textChanged.connect(self.update_data)

    def _populate_ui(self, item):
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.id_spin.setValue(data.get('id', 0))
        self.name_edit.setText(data.get('name', ''))

        self.set_combo_by_data(self.civ_combo, data.get('civilization', 'FIRE'))
        self.set_combo_by_data(self.type_combo, data.get('type', 'CREATURE'))

        self.cost_spin.setValue(data.get('cost', 0))
        self.power_spin.setValue(data.get('power', 0))
        self.races_edit.setText(", ".join(data.get('races', [])))

        # Load Keywords
        kw_data = data.get('keywords', {})
        for k, cb in self.keyword_checks.items():
            is_checked = kw_data.get(k, False)
            cb.setChecked(is_checked)

    def _save_data(self, data):
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
        data['keywords'] = kw_data

    def _get_display_text(self, data):
        return f"{data.get('id', 0)} - {data.get('name', '')}"

    def block_signals_all(self, block):
        self.id_spin.blockSignals(block)
        self.name_edit.blockSignals(block)
        self.civ_combo.blockSignals(block)
        self.type_combo.blockSignals(block)
        self.cost_spin.blockSignals(block)
        self.power_spin.blockSignals(block)
        self.races_edit.blockSignals(block)
        for cb in self.keyword_checks.values():
            cb.blockSignals(block)
