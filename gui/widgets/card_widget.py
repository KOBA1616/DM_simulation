from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette

class CardWidget(QFrame):
    def __init__(self, card_id, card_name, cost, power, civ, tapped=False, parent=None):
        super().__init__(parent)
        self.card_id = card_id
        self.card_name = card_name
        self.cost = cost
        self.power = power
        self.civ = civ
        self.tapped = tapped
        
        self.setFixedSize(100, 140)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setToolTip(f"Name: {self.card_name}\nCost: {self.cost}\nPower: {self.power}\nCiv: {self.civ}")
        
        self.init_ui()
        self.update_style()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        
        # Header (Cost - Name)
        self.header_label = QLabel(f"{self.cost} {self.card_name}")
        self.header_label.setWordWrap(True)
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        font = self.header_label.font()
        font.setBold(True)
        font.setPointSize(8)
        self.header_label.setFont(font)
        layout.addWidget(self.header_label)
        
        layout.addStretch()
        
        # Footer (Power)
        if self.power > 0:
            self.power_label = QLabel(f"BP: {self.power}")
            self.power_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self.power_label)

    def update_style(self):
        # Color based on Civilization
        color_map = {
            "FIRE": "#FFCCCC",
            "WATER": "#CCFFFF",
            "NATURE": "#CCFFCC",
            "LIGHT": "#FFFFCC",
            "DARKNESS": "#E0E0E0"
        }
        bg_color = color_map.get(self.civ, "#FFFFFF")
        
        self.setStyleSheet(f"""
            CardWidget {{
                background-color: {bg_color};
                border: 2px solid {'#555' if not self.tapped else '#000'};
                border-radius: 5px;
            }}
        """)
        
        if self.tapped:
            # Rotate visual representation? 
            # PyQt widgets are hard to rotate in layout.
            # Instead, we change border/color or add "TAPPED" text.
            # For now, let's darken it.
            self.setStyleSheet(f"""
                CardWidget {{
                    background-color: {bg_color};
                    border: 3px solid red;
                    border-radius: 5px;
                }}
            """)

    def set_tapped(self, tapped):
        self.tapped = tapped
        self.update_style()
