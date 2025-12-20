# -*- coding: cp932 -*-
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                             QLineEdit, QComboBox, QHBoxLayout, QLabel, QAbstractItemView)
from PyQt6.QtCore import Qt, QMimeData, QSize
from PyQt6.QtGui import QDrag
from dm_toolkit.gui.localization import tr
import json

class CardSearchWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_cards = []  # List of dicts
        self.filtered_cards = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Filters
        filter_layout = QVBoxLayout()

        # Name search
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel(tr("Name:")))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(tr("Search by name..."))
        self.search_edit.textChanged.connect(self.apply_filter)
        name_layout.addWidget(self.search_edit)
        filter_layout.addLayout(name_layout)

        # Civilization
        civ_layout = QHBoxLayout()
        civ_layout.addWidget(QLabel(tr("Civ:")))
        self.civ_combo = QComboBox()
        self.civ_combo.addItem(tr("All"), "")
        self.civ_combo.addItem(tr("Fire"), "FIRE")
        self.civ_combo.addItem(tr("Water"), "WATER")
        self.civ_combo.addItem(tr("Nature"), "NATURE")
        self.civ_combo.addItem(tr("Light"), "LIGHT")
        self.civ_combo.addItem(tr("Darkness"), "DARKNESS")
        self.civ_combo.currentIndexChanged.connect(self.apply_filter)
        civ_layout.addWidget(self.civ_combo)
        filter_layout.addLayout(civ_layout)

        # Cost
        cost_layout = QHBoxLayout()
        cost_layout.addWidget(QLabel(tr("Cost:")))
        self.cost_edit = QLineEdit()
        self.cost_edit.setPlaceholderText(tr("e.g. 5"))
        self.cost_edit.textChanged.connect(self.apply_filter)
        cost_layout.addWidget(self.cost_edit)
        filter_layout.addLayout(cost_layout)

        layout.addLayout(filter_layout)

        # Results List
        self.result_list = QListWidget()
        self.result_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.result_list.setDragEnabled(True)
        self.result_list.viewport().setAcceptDrops(False)
        self.result_list.setDropIndicatorShown(True)
        self.result_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        layout.addWidget(self.result_list)

    def load_cards(self, cards_data):
        self.all_cards = cards_data
        self.apply_filter()

    def apply_filter(self):
        name_filter = self.search_edit.text().lower()
        civ_filter = self.civ_combo.currentData()
        cost_filter = self.cost_edit.text()

        self.filtered_cards = []
        for card in self.all_cards:
            # Name check
            if name_filter and name_filter not in card.get("name", "").lower():
                continue

            # Civ check
            if civ_filter:
                card_civs = card.get("civilizations", [])
                # If civilizations is list, check if civ_filter is in it
                if isinstance(card_civs, list):
                    if civ_filter not in card_civs:
                        continue
                else:
                    # Legacy or singular
                    if card.get("civilization", "") != civ_filter:
                         # Fallback if card has 'civilization' field
                         if civ_filter not in str(card.get("civilization", "")):
                             continue

            # Cost check
            if cost_filter:
                try:
                    c = int(cost_filter)
                    if int(card.get("cost", 0)) != c:
                        continue
                except ValueError:
                    pass # Ignore invalid cost input

            self.filtered_cards.append(card)

        self.refresh_list()

    def refresh_list(self):
        self.result_list.clear()
        for card in self.filtered_cards:
            item = QListWidgetItem(f"{card.get('name')} ({card.get('cost')})")
            # Store full card name/ID in user role for drag
            item.setData(Qt.ItemDataRole.UserRole, card.get("name"))
            item.setData(Qt.ItemDataRole.UserRole + 1, str(card.get("id")))
            self.result_list.addItem(item)
