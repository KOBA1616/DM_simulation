# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QListWidget, QAbstractItemView
from PyQt6.QtCore import Qt

class DraggableListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            # Handle drop from QListWidget (CardSearchWidget)
            # The default QListWidget MIME data handling is complex to parse manually
            # safely, but since we are copying text items, let's let QListWidget handle it
            # if the formats match.
            # However, CardSearchWidget items carry UserRole data.
            # Standard drop might just copy the display text if not careful.

            # Let's try default implementation first.
            # If it comes from CardSearchWidget, it adds the item.
            # We want to ensure we add just the name or ID string.

            # For simplicity in this iteration, we accept the item as is,
            # and the user sees the name.
            # The underlying logic (save) needs to handle "Name (Cost)" if that's what gets dropped.
            # Actually, CardSearchWidget items have text "Name (Cost)".
            # We probably want just "Name".

            # Custom handling:
            source = event.source()
            if source and source != self:
                # It's an external drop (from search)
                # Iterate selected items in source
                for item in source.selectedItems():
                    # We prefer the raw name stored in UserRole
                    name = item.data(Qt.ItemDataRole.UserRole)
                    if not name:
                         name = item.text() # Fallback
                    self.addItem(name)
                event.accept()
            else:
                # Internal reordering
                super().dropEvent(event)
        else:
            super().dropEvent(event)
