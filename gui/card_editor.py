import csv
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QSpinBox, QPushButton, QMessageBox, QFormLayout
)

class CardEditor(QDialog):
    def __init__(self, csv_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.setWindowTitle("Card Editor")
        self.resize(400, 500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.id_input = QSpinBox()
        self.id_input.setRange(1, 9999)
        # Auto-detect next ID
        self.id_input.setValue(self.get_next_id())
        form.addRow("ID:", self.id_input)

        self.name_input = QLineEdit()
        form.addRow("Name:", self.name_input)

        self.civ_input = QComboBox()
        self.civ_input.addItems(["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE"])
        form.addRow("Civilization:", self.civ_input)

        self.type_input = QComboBox()
        self.type_input.addItems(["CREATURE", "SPELL"])
        form.addRow("Type:", self.type_input)

        self.cost_input = QSpinBox()
        self.cost_input.setRange(0, 99)
        form.addRow("Cost:", self.cost_input)

        self.power_input = QSpinBox()
        self.power_input.setRange(0, 99999)
        self.power_input.setSingleStep(500)
        form.addRow("Power:", self.power_input)

        self.races_input = QLineEdit()
        self.races_input.setPlaceholderText("Comma separated (e.g. Human, Dragon)")
        form.addRow("Races:", self.races_input)

        self.keywords_input = QLineEdit()
        self.keywords_input.setPlaceholderText("Comma separated (e.g. BLOCKER, SPEED_ATTACKER)")
        form.addRow("Keywords:", self.keywords_input)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Card")
        save_btn.clicked.connect(self.save_card)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def get_next_id(self):
        if not os.path.exists(self.csv_path):
            return 1
        max_id = 0
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None) # Skip header
                for row in reader:
                    if row and row[0].isdigit():
                        max_id = max(max_id, int(row[0]))
        except:
            pass
        return max_id + 1

    def save_card(self):
        data = [
            str(self.id_input.value()),
            self.name_input.text().strip(),
            self.civ_input.currentText(),
            self.type_input.currentText(),
            str(self.cost_input.value()),
            str(self.power_input.value()),
            self.races_input.text().strip(),
            self.keywords_input.text().strip()
        ]

        if not data[1]:
            QMessageBox.warning(self, "Error", "Name is required.")
            return

        # Append to CSV
        file_exists = os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["ID","Name","Civilization","Type","Cost","Power","Races","Keywords"])
                writer.writerow(data)
            
            QMessageBox.information(self, "Success", "Card saved successfully!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save card: {e}")
