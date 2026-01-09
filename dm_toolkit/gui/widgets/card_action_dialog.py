# -*- coding: utf-8 -*-
"""
Card Action Dialog - Unified interface for executing card actions with target/parameter selection.
Supports: target selection, numeric input, zone selection, and card quantity specification.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QComboBox,
    QListWidget, QListWidgetItem, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt
import dm_ai_module
from dm_toolkit.gui.localization import tr


class CardActionDialog(QDialog):
    """
    Universal dialog for executing card actions with user input.
    """
    
    def __init__(self, parent=None, game_state=None, card_db=None, action_type="ADD_CARD"):
        super().__init__(parent)
        self.setWindowTitle(tr("Card Action"))
        self.setGeometry(300, 300, 500, 400)
        
        self.gs = game_state
        self.card_db = card_db
        self.action_type = action_type
        self.result_data = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 1. Player Selection
        layout.addWidget(QLabel(tr("Select Player:")))
        self.player_combo = QComboBox()
        self.player_combo.addItem(tr("P0 (Me)"), 0)
        self.player_combo.addItem(tr("P1 (Opponent)"), 1)
        layout.addWidget(self.player_combo)
        
        # 2. Zone Selection (if needed)
        if self.action_type in ["ADD_CARD", "CARD_POSITION"]:
            layout.addWidget(QLabel(tr("Select Zone:")))
            self.zone_combo = QComboBox()
            zones = [
                ("HAND", dm_ai_module.Zone.HAND),
                ("BATTLE_ZONE", dm_ai_module.Zone.BATTLE),
                ("MANA_ZONE", dm_ai_module.Zone.MANA),
                ("SHIELD_ZONE", dm_ai_module.Zone.SHIELD),
                ("GRAVEYARD", dm_ai_module.Zone.GRAVEYARD),
                ("DECK", dm_ai_module.Zone.DECK)
            ]
            for name, z_enum in zones:
                self.zone_combo.addItem(tr(name), z_enum)
            layout.addWidget(self.zone_combo)
        
        # 3. Card Selection (List)
        layout.addWidget(QLabel(tr("Select Card(s):")))
        self.card_list = QListWidget()
        self.card_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        if self.card_db:
            for cid, card in self.card_db.items():
                item = QListWidgetItem(f"{cid}: {card.name}")
                item.setData(Qt.ItemDataRole.UserRole, cid)
                self.card_list.addItem(item)
        
        layout.addWidget(self.card_list)
        
        # 4. Quantity/Parameter Input
        if self.action_type in ["ADD_CARD", "DRAW_CARD"]:
            layout.addWidget(QLabel(tr("Quantity:")))
            self.quantity_spin = QSpinBox()
            self.quantity_spin.setRange(1, 99)
            self.quantity_spin.setValue(1)
            layout.addWidget(self.quantity_spin)
        
        # 5. Buttons
        btn_layout = QHBoxLayout()
        
        ok_btn = QPushButton(tr("OK"))
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton(tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def get_result(self):
        """
        Return dict with action parameters:
        {
            "player_id": int,
            "zone": Zone enum (if applicable),
            "card_ids": list[int],
            "quantity": int (if applicable)
        }
        """
        if getattr(self, 'result_data', None):
            return self.result_data
        
        player_id = self.player_combo.currentData()
        zone = getattr(self, 'zone_combo', None)
        zone_enum = zone.currentData() if zone else None
        
        # Get selected cards
        selected_items = self.card_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, tr("Warning"), tr("Please select at least one card."))
            return None
        
        card_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        quantity = getattr(self, 'quantity_spin', None)
        qty = quantity.value() if quantity else 1
        
        self.result_data = {
            "player_id": player_id,
            "zone": zone_enum,
            "card_ids": card_ids,
            "quantity": qty
        }
        return self.result_data
