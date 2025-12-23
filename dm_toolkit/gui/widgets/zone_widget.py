# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QScrollArea
from PyQt6.QtCore import Qt, pyqtSignal
from .card_widget import CardWidget
from dm_toolkit.gui.localization import get_card_civilization

class ZoneWidget(QWidget):
    card_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_hovered = pyqtSignal(int) # card_id
    action_triggered = pyqtSignal(object) # Action object

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.cards = []
        self.legal_actions = []
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title Label (Vertical)
        self.title_label = QLabel(self.title.replace(" ", "\n"))
        self.title_label.setFixedWidth(40)
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("background-color: #ddd; border: 1px solid #999; font-weight: bold;")
        main_layout.addWidget(self.title_label)
        
        # Scroll Area for Cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(150) 
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.card_container = QWidget()
        self.card_layout = QHBoxLayout(self.card_container)
        self.card_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.card_layout.setContentsMargins(5, 5, 5, 5)
        self.card_layout.setSpacing(5)
        
        self.scroll_area.setWidget(self.card_container)
        main_layout.addWidget(self.scroll_area)

    def set_legal_actions(self, actions):
        self.legal_actions = actions
        # Update existing widgets if possible, but usually update_cards handles recreation
        # If we want live updates without recreation:
        for widget in self.cards:
             if widget.instance_id != -1:
                 relevant = [a for a in self.legal_actions if a.source_instance_id == widget.instance_id]
                 widget.update_legal_actions(relevant)

    def update_cards(self, card_data_list, card_db, civ_map=None, legal_actions=None):
        # Update cached legal actions if provided
        if legal_actions is not None:
            self.legal_actions = legal_actions

        # Clear existing
        for i in reversed(range(self.card_layout.count())):
            item = self.card_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        self.cards = []

        # Check for Deck Bundle Visualization
        is_deck = "Deck" in self.title or "デッキ" in self.title
        is_shield = "Shield" in self.title or "シールド" in self.title

        if (is_deck or is_shield) and card_data_list:
            # Single Bundle Representation
            count = len(card_data_list)
            # Use ID 0 (Back of Card)
            display_name = f"Deck ({count})" if is_deck else f"Shield ({count})"
            # Pass is_face_down=True
            widget = CardWidget(0, display_name, 0, 0, "COLORLESS", False, -1, None, True)
            # Clicking emits signal with ID 0
            widget.clicked.connect(lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))
            widget.hovered.connect(self.card_hovered.emit)
            self.card_layout.addWidget(widget)
            self.cards.append(widget)
            return

        # Normal Visualization
        for c_data in card_data_list:
            # c_data: (card_id, is_tapped) or just card_id
            cid = c_data['id']
            tapped = c_data.get('tapped', False)
            instance_id = c_data.get('instance_id', -1)
            
            # Filter actions for this card
            relevant_actions = []
            if instance_id != -1:
                relevant_actions = [a for a in self.legal_actions if a.source_instance_id == instance_id]

            if cid in card_db:
                card_def = card_db[cid]
                civ = get_card_civilization(card_def)
                
                widget = CardWidget(
                    cid, card_def.name, card_def.cost, card_def.power, 
                    civ, tapped, instance_id,
                    legal_actions=relevant_actions
                )
                widget.clicked.connect(lambda i_id, c_id=cid: self.card_clicked.emit(c_id, i_id))
                widget.hovered.connect(self.card_hovered.emit)
                widget.action_triggered.connect(self.action_triggered.emit)
                self.card_layout.addWidget(widget)
                self.cards.append(widget)
            else:
                # Unknown/Masked
                # Pass is_face_down=True
                widget = CardWidget(0, "???", 0, 0, "COLORLESS", False, instance_id, None, True, legal_actions=relevant_actions)
                widget.clicked.connect(lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))
                widget.hovered.connect(self.card_hovered.emit)
                widget.action_triggered.connect(self.action_triggered.emit)
                self.card_layout.addWidget(widget)

    def set_card_selected(self, instance_id, selected):
        for widget in self.cards:
            if widget.instance_id == instance_id:
                widget.set_selected(selected)
                return
