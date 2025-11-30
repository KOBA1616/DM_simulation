from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class CardDetailPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.name_label = QLabel("Card Name")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)
        
        self.info_label = QLabel("Cost: - | Power: - | Civ: -")
        layout.addWidget(self.info_label)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText("Card effects will appear here...")
        layout.addWidget(self.text_area)

    def update_card(self, card_data, civ_map=None):
        if not card_data:
            self.name_label.setText("Unknown Card")
            self.info_label.setText("Cost: ? | Power: ? | Civ: ?")
            self.text_area.setText("")
            return

        civ = "COLORLESS"
        if civ_map and card_data.id in civ_map:
            civ = civ_map[card_data.id]
            
        self.name_label.setText(card_data.name)
        self.info_label.setText(f"Cost: {card_data.cost} | Power: {card_data.power} | Civ: {civ}")
        
        self.text_area.setText(f"ID: {card_data.id}\nType: {card_data.type}")
