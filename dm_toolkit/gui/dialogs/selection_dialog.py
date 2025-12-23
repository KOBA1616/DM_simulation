# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout,
    QLabel, QAbstractItemView, QListWidgetItem, QWidget
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon

class CardSelectionDialog(QDialog):
    """
    A generic dialog for selecting cards or actions.
    Supports single or multiple selection.
    """
    def __init__(self, title, instruction, items, min_selection=1, max_selection=1, parent=None, card_db=None):
        """
        items: List of objects to display. Can be:
               - Dict with 'name', 'desc', 'card_id', 'id' (for effect selection)
               - Card objects (if adapting from card list)
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)
        self.min_selection = min_selection
        self.max_selection = max_selection
        self.card_db = card_db
        self.selected_indices = []

        layout = QVBoxLayout(self)

        # Instruction Label
        lbl = QLabel(instruction)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(lbl)

        # List Widget
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(32, 32))

        if max_selection > 1:
            self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        else:
            self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.populate_list(items)
        layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept_selection)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        # If selection is mandatory (min >= 1), user shouldn't be able to just close/cancel easily
        # unless the game flow allows it. But for a dialog, Cancel usually means "Cancel Action" if possible.

        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        # Update button state initially
        self.list_widget.itemSelectionChanged.connect(self.validate_selection)
        self.validate_selection()

    def populate_list(self, items):
        for idx, item in enumerate(items):
            # Determine display text based on item type
            display_text = "Unknown Item"
            card_id = -1

            if isinstance(item, dict):
                # Expecting pending effect dict
                name = item.get('source_name', 'Unknown')
                desc = item.get('description', '')
                card_id = item.get('card_id', -1)
                display_text = f"{name}\n{desc}"
            elif hasattr(item, 'card_id'):
                # Card Object
                card_id = item.card_id
                if self.card_db:
                    cdef = self.card_db.get(card_id)
                    name = cdef.name if cdef else "Unknown"
                    display_text = f"{name} (Cost: {cdef.cost})"
                else:
                    display_text = f"Card {card_id}"

            list_item = QListWidgetItem(display_text)
            list_item.setData(Qt.ItemDataRole.UserRole, idx)

            # TODO: If we have card images or icons, set them here
            # list_item.setIcon(...)

            self.list_widget.addItem(list_item)

    def validate_selection(self):
        selected = self.list_widget.selectedItems()
        count = len(selected)
        is_valid = self.min_selection <= count <= self.max_selection
        self.ok_btn.setEnabled(is_valid)

        # Update OK button text to show count
        if self.max_selection > 1:
            self.ok_btn.setText(f"OK ({count}/{self.max_selection})")
        else:
            self.ok_btn.setText("OK")

    def accept_selection(self):
        selected_items = self.list_widget.selectedItems()
        self.selected_indices = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        self.accept()

    def get_selected_indices(self):
        return self.selected_indices
