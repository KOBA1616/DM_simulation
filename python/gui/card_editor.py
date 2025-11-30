import csv
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QSpinBox, QPushButton, QMessageBox, QFormLayout, QFrame, QWidget, QCheckBox, QGridLayout, QScrollArea
)
from gui.widgets.card_widget import CardWidget

class CardEditor(QDialog):
    def __init__(self, csv_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.setWindowTitle("Card Editor")
        self.resize(600, 500)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Left: Form (Scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form = QFormLayout()

        self.id_input = QSpinBox()
        self.id_input.setRange(1, 9999)
        # Auto-detect next ID
        self.id_input.setValue(self.get_next_id())
        form.addRow("ID:", self.id_input)

        self.name_input = QLineEdit()
        self.name_input.textChanged.connect(self.update_preview)
        form.addRow("Name:", self.name_input)

        self.civ_input = QComboBox()
        self.civ_input.addItems(["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"])
        self.civ_input.currentTextChanged.connect(self.update_preview)
        form.addRow("Civilization:", self.civ_input)

        self.type_input = QComboBox()
        self.type_input.addItems(["CREATURE", "SPELL", "EVOLUTION_CREATURE"])
        form.addRow("Type:", self.type_input)

        self.cost_input = QSpinBox()
        self.cost_input.setRange(0, 99)
        self.cost_input.valueChanged.connect(self.update_preview)
        form.addRow("Cost:", self.cost_input)

        self.power_input = QSpinBox()
        self.power_input.setRange(0, 99999)
        self.power_input.setSingleStep(500)
        self.power_input.valueChanged.connect(self.update_preview)
        form.addRow("Power:", self.power_input)

        self.races_input = QLineEdit()
        self.races_input.setPlaceholderText("Semicolon separated (e.g. Human; Dragon)")
        form.addRow("Races:", self.races_input)

        # Keywords Checkboxes
        keywords_label = QLabel("Keywords:")
        form.addRow(keywords_label)
        
        self.keywords_layout = QGridLayout()
        self.keyword_checkboxes = {}
        keywords_list = [
            "BLOCKER", "SPEED_ATTACKER", "SHIELD_TRIGGER", "SLAYER",
            "DOUBLE_BREAKER", "TRIPLE_BREAKER", "POWER_ATTACKER",
            "EVOLUTION", "CIP", "AT_ATTACK", "AT_BLOCK", 
            "AT_START_OF_TURN", "AT_END_OF_TURN", "ON_DESTROY",
            "G_STRIKE", "MACH_FIGHTER", "REVOLUTION_CHANGE", "G_ZERO"
        ]
        
        for i, kw in enumerate(keywords_list):
            cb = QCheckBox(kw)
            self.keyword_checkboxes[kw] = cb
            self.keywords_layout.addWidget(cb, i // 2, i % 2)
            
        form.addRow(self.keywords_layout)

        form_layout.addLayout(form)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Card")
        save_btn.clicked.connect(self.save_card)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        form_layout.addLayout(btn_layout)
        
        scroll_area.setWidget(form_widget)
        main_layout.addWidget(scroll_area, 2) # Give form more space
        
        # Right: Preview
        preview_layout = QVBoxLayout()
        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)
        
        self.preview_container = QWidget()
        self.preview_container_layout = QVBoxLayout(self.preview_container)
        self.preview_card = None
        
        preview_layout.addWidget(self.preview_container)
        preview_layout.addStretch()
        
        main_layout.addLayout(preview_layout, 1)
        
        self.update_preview()

    def update_preview(self):
        # Clear old preview
        for i in reversed(range(self.preview_container_layout.count())):
            item = self.preview_container_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        name = self.name_input.text() or "New Card"
        cost = self.cost_input.value()
        power = self.power_input.value()
        civ = self.civ_input.currentText()
        
        self.preview_card = CardWidget(0, name, cost, power, civ)
        self.preview_container_layout.addWidget(self.preview_card)

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
        # Collect keywords
        selected_keywords = []
        for kw, cb in self.keyword_checkboxes.items():
            if cb.isChecked():
                selected_keywords.append(kw)
        keywords_str = ";".join(selected_keywords)

        # Handle Races (replace commas with semicolons just in case, remove quotes)
        races_str = self.races_input.text().strip().replace(",", ";").replace('"', '')

        data = [
            str(self.id_input.value()),
            self.name_input.text().strip().replace(",", "").replace('"', ''),
            self.civ_input.currentText(),
            self.type_input.currentText(),
            str(self.cost_input.value()),
            str(self.power_input.value()),
            races_str,
            keywords_str
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
