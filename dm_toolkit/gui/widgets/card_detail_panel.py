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
        # Use civ from card_data if available (it is exposed now)
        if hasattr(card_data, 'civilization'):
             civ = str(card_data.civilization).split('.')[-1]
        elif civ_map and card_data.id in civ_map:
            civ = civ_map[card_data.id]
            
        self.name_label.setText(card_data.name)
        self.info_label.setText(f"Cost: {card_data.cost} | Power: {card_data.power} | Civ: {civ}")
        
        text = f"ID: {card_data.id}\n"
        text += f"Type: {str(card_data.type).split('.')[-1]}\n"
        
        if hasattr(card_data, 'races') and card_data.races:
            text += f"Races: {', '.join(card_data.races)}\n"
            
        if hasattr(card_data, 'keywords'):
            k = card_data.keywords
            kws = []
            if k.blocker: kws.append("Blocker")
            if k.speed_attacker: kws.append("Speed Attacker")
            if k.slayer: kws.append("Slayer")
            if k.double_breaker: kws.append("W-Breaker")
            if k.triple_breaker: kws.append("T-Breaker")
            if k.power_attacker: kws.append(f"Power Attacker +{card_data.power_attacker_bonus}")
            if k.shield_trigger: kws.append("Shield Trigger")
            if k.g_strike: kws.append("G-Strike")
            if k.mach_fighter: kws.append("Mach Fighter")
            if k.revolution_change: kws.append("Revolution Change")
            if k.g_zero: kws.append("G-Zero")
            if k.evolution: kws.append("Evolution")
            if k.cip: kws.append("CIP")
            if k.at_attack: kws.append("At Attack")
            if k.at_block: kws.append("At Block")
            if k.at_start_of_turn: kws.append("Start of Turn")
            if k.at_end_of_turn: kws.append("End of Turn")
            if k.destruction: kws.append("On Destroy")
            
            if kws:
                text += f"Keywords: {', '.join(kws)}\n"
        
        self.text_area.setText(text)
