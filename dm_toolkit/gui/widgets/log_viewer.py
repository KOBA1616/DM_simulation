# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QListWidget
from PyQt6.QtCore import Qt

class LogViewer(QListWidget):
    """
    A widget for displaying game logs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LogViewer")
        # Optional: set selection mode or other properties
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

    def log_message(self, msg: str) -> None:
        """Appends a message to the log and scrolls to bottom."""
        self.addItem(msg)
        self.scrollToBottom()

    def clear_logs(self) -> None:
        """Clears all log items."""
        self.clear()
