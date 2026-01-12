# -*- coding: utf-8 -*-
from typing import List, Dict, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter
from PyQt6.QtCore import pyqtSignal, Qt

from dm_toolkit.gui.widgets.zone_widget import ZoneWidget
from dm_toolkit.gui.i18n import tr

class GameBoard(QWidget):
    """
    Manages the game board layout including player zones.
    """
    action_triggered = pyqtSignal(object)  # Emits action object
    card_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_double_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_hovered = pyqtSignal(int) # card_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)

        # P1 Zones (Opponent)
        self.p1_zones = QWidget()
        self.p1_layout = QVBoxLayout(self.p1_zones)
        self.p1_hand = ZoneWidget(tr("P1 Hand"))
        self.p1_mana = ZoneWidget(tr("P1 Mana"))
        self.p1_graveyard = ZoneWidget(tr("P1 Graveyard"))
        self.p1_battle = ZoneWidget(tr("P1 Battle Zone"))
        self.p1_shield = ZoneWidget(tr("P1 Shield Zone"))
        self.p1_deck_zone = ZoneWidget(tr("P1 Deck"))

        self.p1_layout.addWidget(self.p1_hand)
        p1_row2 = QHBoxLayout()
        p1_row2.addWidget(self.p1_mana, stretch=3)
        p1_row2.addWidget(self.p1_shield, stretch=2)
        p1_row2.addWidget(self.p1_graveyard, stretch=1)
        self.p1_layout.addLayout(p1_row2)

        p1_battle_row = QHBoxLayout()
        p1_battle_row.addWidget(self.p1_battle, stretch=5)
        p1_battle_row.addWidget(self.p1_deck_zone, stretch=1)
        self.p1_layout.addLayout(p1_battle_row)

        # P0 Zones (Self)
        self.p0_zones = QWidget()
        self.p0_layout = QVBoxLayout(self.p0_zones)
        self.p0_battle = ZoneWidget(tr("P0 Battle Zone"))
        self.p0_deck_zone = ZoneWidget(tr("P0 Deck"))
        self.p0_shield = ZoneWidget(tr("P0 Shield Zone"))
        self.p0_mana = ZoneWidget(tr("P0 Mana"))
        self.p0_graveyard = ZoneWidget(tr("P0 Graveyard"))
        self.p0_hand = ZoneWidget(tr("P0 Hand"))

        p0_battle_row = QHBoxLayout()
        p0_battle_row.addWidget(self.p0_battle, stretch=5)
        p0_battle_row.addWidget(self.p0_deck_zone, stretch=1)
        self.p0_layout.addLayout(p0_battle_row)

        p0_row2 = QHBoxLayout()
        p0_row2.addWidget(self.p0_mana, stretch=3)
        p0_row2.addWidget(self.p0_shield, stretch=2)
        p0_row2.addWidget(self.p0_graveyard, stretch=1)
        self.p0_layout.addLayout(p0_row2)
        self.p0_layout.addWidget(self.p0_hand)

        # Connect Signals
        self._connect_zone(self.p0_hand)
        self._connect_zone(self.p0_mana)
        self._connect_zone(self.p0_battle)
        self._connect_zone(self.p0_graveyard)
        self._connect_zone(self.p0_shield) # Shield usually not clickable for actions but maybe hover

        # Opponent zones usually just hover
        for z in [self.p1_hand, self.p1_mana, self.p1_battle, self.p1_shield, self.p1_graveyard]:
            z.card_hovered.connect(self.card_hovered.emit)

        # Splitter
        self.board_splitter = QSplitter(Qt.Orientation.Vertical)
        self.board_splitter.addWidget(self.p1_zones)
        self.board_splitter.addWidget(self.p0_zones)
        self.layout_main.addWidget(self.board_splitter)

    def _connect_zone(self, zone: ZoneWidget):
        zone.action_triggered.connect(self.action_triggered.emit)
        zone.card_clicked.connect(self.card_clicked.emit)
        zone.card_double_clicked.connect(self.card_double_clicked.emit)
        zone.card_hovered.connect(self.card_hovered.emit)

    def update_state(self, p0_data: Any, p1_data: Any, card_db: Any, legal_actions: List[Any], god_view: bool = False):
        """
        Updates all zones based on player data objects (EngineCompat.get_player result).
        """
        def convert_zone(zone_cards: List[Any], hide: bool=False) -> List[Dict[str, Any]]:
            if hide: return [{'id': -1, 'tapped': getattr(c, 'is_tapped', False), 'instance_id': getattr(c, 'instance_id', -1)} for c in zone_cards]
            return [{'id': getattr(c, 'card_id', -1), 'tapped': getattr(c, 'is_tapped', False), 'instance_id': getattr(c, 'instance_id', -1)} for c in zone_cards]

        # P0 Update (Self)
        self.p0_hand.update_cards(convert_zone(p0_data.hand), card_db, legal_actions=legal_actions)
        self.p0_mana.update_cards(convert_zone(p0_data.mana_zone), card_db, legal_actions=legal_actions)
        self.p0_battle.update_cards(convert_zone(p0_data.battle_zone), card_db, legal_actions=legal_actions)
        self.p0_shield.update_cards(convert_zone(p0_data.shield_zone), card_db, legal_actions=legal_actions)
        self.p0_graveyard.update_cards(convert_zone(p0_data.graveyard), card_db, legal_actions=legal_actions)
        self.p0_deck_zone.update_cards(convert_zone(p0_data.deck, hide=True), card_db, legal_actions=legal_actions)

        # P1 Update (Opponent)
        self.p1_hand.update_cards(convert_zone(p1_data.hand, hide=not god_view), card_db)
        self.p1_mana.update_cards(convert_zone(p1_data.mana_zone), card_db)
        self.p1_battle.update_cards(convert_zone(p1_data.battle_zone), card_db)
        self.p1_shield.update_cards(convert_zone(p1_data.shield_zone, hide=not god_view), card_db)
        self.p1_graveyard.update_cards(convert_zone(p1_data.graveyard), card_db)
        self.p1_deck_zone.update_cards(convert_zone(p1_data.deck, hide=True), card_db)

    def set_selection_mode(self, selected_targets: List[int]):
        """
        Highlights selected cards across zones.
        """
        zones = [
            self.p0_hand, self.p0_mana, self.p0_battle, self.p0_shield, self.p0_graveyard,
            self.p1_hand, self.p1_mana, self.p1_battle, self.p1_shield, self.p1_graveyard
        ]
        for zone in zones:
            for target_id in selected_targets:
                zone.set_card_selected(target_id, True)
