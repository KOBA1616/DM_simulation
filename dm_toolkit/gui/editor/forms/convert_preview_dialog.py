# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt
import json
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor import text_generator


class ConvertPreviewDialog(QDialog):
    """Shows original Action JSON and converted Command JSON side-by-side.

    Returns user's choice via exec_():
      Accepted -> QDialog.Accepted (use converted)
      Rejected -> QDialog.Rejected (keep action)
      Cancel -> QDialog.DialogCode(0) (cancel save)
    """
    def __init__(self, parent, action_data: dict, converted: dict):
        super().__init__(parent)
        # Safe defaults for attributes used by external checkers/tests
        self.action_text = None
        self.conv_text = None
        self.action_preview = None
        self.conv_preview = None
        self.use_btn = None
        self.keep_btn = None
        self.cancel_btn = None
        self.setWindowTitle(tr("Conversion Preview"))
        self.resize(800, 480)

        layout = QVBoxLayout(self)

        hint = QLabel(tr("The editor attempted to convert this legacy Action to a Command. Review the result."))
        layout.addWidget(hint)

        panes = QHBoxLayout()
        self.action_text = QTextEdit()
        self.action_text.setReadOnly(True)
        self.action_text.setPlainText(json.dumps(action_data, indent=2, ensure_ascii=False))

        self.conv_text = QTextEdit()
        self.conv_text.setReadOnly(True)
        self.conv_text.setPlainText(json.dumps(converted, indent=2, ensure_ascii=False))

        panes.addWidget(self.action_text)
        panes.addWidget(self.conv_text)
        layout.addLayout(panes)

        # Generated human-readable previews
        previews = QHBoxLayout()
        self.action_preview = QTextEdit()
        self.action_preview.setReadOnly(True)
        try:
            a_text = text_generator.CardTextGenerator._format_action(action_data)
        except Exception:
            a_text = ""
        self.action_preview.setPlainText(a_text)

        self.conv_preview = QTextEdit()
        self.conv_preview.setReadOnly(True)
        try:
            # Prefer using _format_command for command-shaped dicts
            c_text = text_generator.CardTextGenerator._format_command(converted)
        except Exception:
            try:
                c_text = text_generator.CardTextGenerator._format_action(converted)
            except Exception:
                c_text = ""
        self.conv_preview.setPlainText(c_text)

        previews.addWidget(self.action_preview)
        previews.addWidget(self.conv_preview)
        layout.addLayout(previews)

        btns = QHBoxLayout()
        self.use_btn = QPushButton(tr("Use Converted Command"))
        self.keep_btn = QPushButton(tr("Keep as Action"))
        self.cancel_btn = QPushButton(tr("Cancel"))

        self.use_btn.clicked.connect(self.accept)
        self.keep_btn.clicked.connect(self.reject)
        self.cancel_btn.clicked.connect(self.close)

        btns.addWidget(self.use_btn)
        btns.addWidget(self.keep_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

    def keyPressEvent(self, ev):
        # Escape should close as cancel
        if ev.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(ev)
