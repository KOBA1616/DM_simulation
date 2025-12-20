# -*- coding: cp932 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from dm_toolkit.gui.localization import tr

class CardDetailPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.name_label = QLabel(tr("Card Name"))
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)
        
        self.info_label = QLabel(f"{tr('Cost')}: - | {tr('Power')}: - | {tr('Civ')}: -")
        layout.addWidget(self.info_label)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText(tr("Card effects will appear here..."))
        layout.addWidget(self.text_area)

    def update_card(self, card_data, civ_map=None):
        if not card_data:
            self.name_label.setText(tr("Unknown Card"))
            self.info_label.setText(f"{tr('Cost')}: ? | {tr('Power')}: ? | {tr('Civ')}: ?")
            self.text_area.setText("")
            return

        civ = "COLORLESS"
        # Prioritize civilizations vector (multi-civ)
        if hasattr(card_data, 'civilizations') and card_data.civilizations:
             civs = [tr(str(c).split('.')[-1]) for c in card_data.civilizations]
             civ = "/".join(civs)
        # Fallback to single civilization property
        elif hasattr(card_data, 'civilization'):
             civ = tr(str(card_data.civilization).split('.')[-1])
        # Legacy fallback (should be removed eventually, kept for safety)
        elif civ_map and card_data.id in civ_map:
            civ = tr(civ_map[card_data.id])

        self.name_label.setText(card_data.name)
        self.info_label.setText(f"{tr('Cost')}: {card_data.cost} | {tr('Power')}: {card_data.power} | {tr('Civ')}: {civ}")
        
        text = f"ID: {card_data.id}\n"
        text += f"{tr('Type')}: {str(card_data.type).split('.')[-1]}\n"
        
        if hasattr(card_data, 'races') and card_data.races:
            text += f"{tr('Races')}: {', '.join(card_data.races)}\n"
            
        if hasattr(card_data, 'keywords'):
            k = card_data.keywords
            kws = []
            if k.blocker: kws.append(tr("Blocker"))
            if k.speed_attacker: kws.append(tr("Speed Attacker"))
            if k.slayer: kws.append(tr("Slayer"))
            if k.double_breaker: kws.append(tr("W-Breaker"))
            if k.triple_breaker: kws.append(tr("T-Breaker"))
            if k.power_attacker: kws.append(f"{tr('Power Attacker')} +{card_data.power_attacker_bonus}")
            if k.shield_trigger: kws.append(tr("Shield Trigger"))
            if k.g_strike: kws.append(tr("G-Strike"))
            if k.mach_fighter: kws.append(tr("Mach Fighter"))
            if k.revolution_change: kws.append(tr("Revolution Change"))
            if k.g_zero: kws.append(tr("G-Zero"))
            if k.evolution: kws.append(tr("Evolution"))
            if k.hyper_energy: kws.append(tr("Hyper Energy"))
            if k.just_diver: kws.append(tr("Just Diver"))
            if k.cip: kws.append(tr("CIP"))
            if k.at_attack: kws.append(tr("At Attack"))
            if k.at_block: kws.append(tr("At Block"))
            if k.at_start_of_turn: kws.append(tr("Start of Turn"))
            if k.at_end_of_turn: kws.append(tr("End of Turn"))
            if k.destruction: kws.append(tr("On Destroy"))
            
            if kws:
                text += f"{tr('Keywords')}: {', '.join(kws)}\n"
        
        self.text_area.setText(text)
