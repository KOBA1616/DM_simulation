# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QListWidget, QMenu
try:
    from PyQt6.QtWidgets import QAction
except Exception:
    from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication

class LogViewer(QListWidget):
    """
    A widget for displaying game logs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LogViewer")
        # Allow multi-selection so users can select multiple log lines to copy
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # Enable custom context menu for copy
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        copy_act = QAction("Copy", self)
        # PyQt5 had QKeySequence.Copy; PyQt6 exposes StandardKey or may omit the attribute.
        # Use string-based fallback to ensure compatibility across environments.
        try:
            copy_act.setShortcut(QKeySequence.Copy)
        except Exception:
            try:
                copy_act.setShortcut(QKeySequence("Ctrl+C"))
            except Exception:
                pass
        copy_act.triggered.connect(self.copy_selected)
        menu.addAction(copy_act)
        # Add select-all for convenience
        sa = QAction("Select All", self)
        sa.triggered.connect(self.selectAll)
        menu.addAction(sa)
        menu.exec(self.mapToGlobal(pos))

    def copy_selected(self):
        items = self.selectedItems()
        if not items:
            return
        text = "\n".join([it.text() for it in items])
        try:
            cb = QApplication.clipboard()
            cb.setText(text)
        except Exception:
            pass

    def keyPressEvent(self, event):
        # Support Ctrl+C copying
        # Robust Ctrl+C detection without relying on QKeySequence.Copy symbol
        try:
            mods = event.modifiers()
            key = event.key()
            if (mods & Qt.KeyboardModifier.ControlModifier) and (key == Qt.Key.Key_C):
                self.copy_selected()
                return
        except Exception:
            pass
        super().keyPressEvent(event)

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
