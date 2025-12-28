# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QScrollArea
from PyQt6.QtCore import Qt, pyqtSignal
from .card_widget import CardWidget
from dm_toolkit.gui.localization import get_card_civilization
from dm_toolkit.action_to_command import ActionToCommand

class ZoneWidget(QWidget):
    card_clicked = pyqtSignal(int, int) # card_id, instance_id
    card_hovered = pyqtSignal(int) # card_id
    action_triggered = pyqtSignal(object) # CommandDefDict (dict)

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

    def _get_action_source_id(self, action):
        """Helper to safely get source_instance_id from dict or object"""
        if isinstance(action, dict):
            return action.get('source_instance_id', -1)
        return getattr(action, 'source_instance_id', -1)

    def set_legal_actions(self, actions):
        """
        Sets legal actions. Actions can be legacy Action objects or new Command objects.
        """
        self.legal_actions = actions
        for widget in self.cards:
             if widget.instance_id != -1:
                 relevant = [a for a in self.legal_actions if self._get_action_source_id(a) == widget.instance_id]
                 widget.update_legal_actions(relevant)

    def update_cards(self, card_data_list, card_db, civ_map=None, legal_actions=None):
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
            count = len(card_data_list)
            display_name = f"Deck ({count})" if is_deck else f"Shield ({count})"
            widget = CardWidget(0, display_name, 0, 0, "COLORLESS", False, -1, None, True)
            widget.clicked.connect(lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))
            widget.hovered.connect(self.card_hovered.emit)
            self.card_layout.addWidget(widget)
            self.cards.append(widget)
            return

        # Normal Visualization
        for c_data in card_data_list:
            cid = c_data['id']
            tapped = c_data.get('tapped', False)
            instance_id = c_data.get('instance_id', -1)
            
            relevant_actions = []
            if instance_id != -1:
                relevant_actions = [a for a in self.legal_actions if self._get_action_source_id(a) == instance_id]

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
                widget.action_triggered.connect(self._handle_action_triggered)
                self.card_layout.addWidget(widget)
                self.cards.append(widget)
            else:
                widget = CardWidget(0, "???", 0, 0, "COLORLESS", False, instance_id, None, True, legal_actions=relevant_actions)
                widget.clicked.connect(lambda i_id, c_id=0: self.card_clicked.emit(c_id, i_id))
                widget.hovered.connect(self.card_hovered.emit)
                widget.action_triggered.connect(self._handle_action_triggered)
                self.card_layout.addWidget(widget)

    def set_card_selected(self, instance_id, selected):
        for widget in self.cards:
            if widget.instance_id == instance_id:
                widget.set_selected(selected)
                return

    def _handle_action_triggered(self, action_obj):
        """
        Intercepts action signal, converts to CommandDef via ActionToCommand if necessary, and emits.
        """
        # If it's already a command structure (e.g. dict with proper type), pass it
        if isinstance(action_obj, dict) and "command_def" in action_obj: # Assuming wrapper
             self.action_triggered.emit(action_obj)
             return

        # Use the unified converter
        cmd = ActionToCommand.map_action(action_obj)
        self.action_triggered.emit(cmd)
