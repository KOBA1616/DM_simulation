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

    def update_from_history(self, history: list, start_index: int, gs, card_db) -> int:
        """
        Updates the log with new commands from history starting at start_index.
        Returns the new length of history processed.
        """
        from dm_toolkit.gui.utils.command_describer import describe_command
        try:
            current_len = len(history)
        except:
            current_len = 0

        if current_len > start_index:
            for i in range(start_index, current_len):
                cmd = history[i]
                try:
                    desc = describe_command(cmd, gs, card_db)
                except:
                    desc = str(cmd)
                self.log_message(desc)
            return current_len
        return start_index
