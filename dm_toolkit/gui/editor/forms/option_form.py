# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm
from dm_toolkit.gui.localization import tr

class OptionForm(BaseEditForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Safe defaults
        self.label = getattr(self, 'label', None)
        try:
            self.setup_ui()
        except Exception:
            pass

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.label = QLabel(tr("Option"))
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.label)

        info_label = QLabel(tr("This is a container for conditional branches or multiple choice options."))
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

    def set_data(self, item):
        super().set_data(item)
        self.label.setText(item.text())
