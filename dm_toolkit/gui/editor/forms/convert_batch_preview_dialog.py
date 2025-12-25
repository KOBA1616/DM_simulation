# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QDialogButtonBox, QTextEdit, QSplitter
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
import json
from dm_toolkit.gui.localization import tr

class ConvertBatchPreviewDialog(QDialog):
    def __init__(self, parent, preview_items):
        super().__init__(parent)
        self.setWindowTitle(tr("Preview Conversions"))
        self.resize(640, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr("The following actions will be converted to Commands. Review warnings before applying.")))

        self.list_widget = QListWidget(self)
        for p in preview_items:
            text = f"{p['label']} — {p['path']}"
            item = QListWidgetItem(text)
            if p.get('warning'):
                # Use a simple warning prefix; icon may not be available cross-platform
                item.setText("⚠️ " + item.text())
                item.setData(Qt.ItemDataRole.UserRole, True)
            # Store full preview dict for later inspection
            item.setData(Qt.ItemDataRole.UserRole + 1, p)
            self.list_widget.addItem(item)

        # Splitter: left shows original Action JSON, right shows converted Command JSON
        self.splitter = QSplitter(self)
        self.action_view = QTextEdit(self)
        self.action_view.setReadOnly(True)
        self.command_view = QTextEdit(self)
        self.command_view.setReadOnly(True)
        self.splitter.addWidget(self.action_view)
        self.splitter.addWidget(self.command_view)
        self.splitter.setSizes([320, 320])

        # Update preview when selection changes
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)

        layout.addWidget(self.list_widget)
        layout.addWidget(self.splitter)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Expand height to show enough items
        self.setLayout(layout)

    def _on_selection_changed(self, current, previous):
        if current is None:
            self.action_view.clear()
            self.command_view.clear()
            return

        preview = current.data(Qt.ItemDataRole.UserRole + 1) or {}
        # Try to show original action if present, else show label
        action_obj = preview.get('action') or preview.get('original_action') or preview.get('act_data')
        cmd_obj = preview.get('cmd_data') or preview.get('command') or {}

        try:
            action_text = json.dumps(action_obj, ensure_ascii=False, indent=2) if action_obj else current.text()
        except Exception:
            action_text = str(action_obj)

        try:
            cmd_text = json.dumps(cmd_obj, ensure_ascii=False, indent=2) if cmd_obj else "(No converted command)"
        except Exception:
            cmd_text = str(cmd_obj)

        self.action_view.setPlainText(action_text)
        self.command_view.setPlainText(cmd_text)
*** End Patch