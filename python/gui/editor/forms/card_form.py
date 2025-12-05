from PyQt6.QtWidgets import QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox, QCheckBox, QLabel, QGridLayout, QGroupBox
from PyQt6.QtCore import Qt
from gui.localization import tr

class CardEditForm(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item = None
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 9999)
        layout.addRow(tr("ID"), self.id_spin)

        self.name_edit = QLineEdit()
        layout.addRow(tr("Name"), self.name_edit)

        self.civ_combo = QComboBox()
        self.civ_combo.addItems(["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"])
        layout.addRow(tr("Civilization"), self.civ_combo)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["CREATURE", "SPELL", "EVOLUTION_CREATURE"])
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

        # Connect signals
        self.id_spin.valueChanged.connect(self.update_data)
        self.name_edit.textChanged.connect(self.update_data)
        self.civ_combo.currentTextChanged.connect(self.update_data)
        self.type_combo.currentTextChanged.connect(self.update_data)
        self.cost_spin.valueChanged.connect(self.update_data)
        self.power_spin.valueChanged.connect(self.update_data)
        self.races_edit.textChanged.connect(self.update_data)

    def set_data(self, item):
        self.current_item = item
        data = item.data(Qt.ItemDataRole.UserRole + 2)

        self.block_signals(True)
        self.id_spin.setValue(data.get('id', 0))
        self.name_edit.setText(data.get('name', ''))
        self.civ_combo.setCurrentText(data.get('civilization', 'FIRE'))
        self.type_combo.setCurrentText(data.get('type', 'CREATURE'))
        self.cost_spin.setValue(data.get('cost', 0))
        self.power_spin.setValue(data.get('power', 0))
        self.races_edit.setText(", ".join(data.get('races', [])))
        self.block_signals(False)

    def update_data(self):
        if not self.current_item: return
        data = self.current_item.data(Qt.ItemDataRole.UserRole + 2)

        data['id'] = self.id_spin.value()
        data['name'] = self.name_edit.text()
        data['civilization'] = self.civ_combo.currentText()
        data['type'] = self.type_combo.currentText()
        data['cost'] = self.cost_spin.value()
        data['power'] = self.power_spin.value()
        races_str = self.races_edit.text()
        data['races'] = [r.strip() for r in races_str.split(',') if r.strip()]

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
