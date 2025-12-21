# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
from dm_toolkit.gui.localization import tr

class BaseWizardDialog(QDialog):
    """
    Base class for mechanic wizards.
    Provides a standard layout with Title, Description, Content Area, and OK/Cancel buttons.
    """
    def __init__(self, parent=None, title="Wizard"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result_data = None
        self.setup_base_ui()

    def setup_base_ui(self):
        self.main_layout = QVBoxLayout(self)

        # Title / Description
        self.lbl_title = QLabel(self.windowTitle())
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.main_layout.addWidget(self.lbl_title)

        self.lbl_desc = QLabel("")
        self.lbl_desc.setWordWrap(True)
        self.main_layout.addWidget(self.lbl_desc)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line)

        # Content Area (Subclasses add widgets here)
        self.content_layout = QVBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch()

        self.btn_ok = QPushButton(tr("OK")) # Should be Localized "OK"
        self.btn_ok.clicked.connect(self.accept)
        self.button_layout.addWidget(self.btn_ok)

        self.btn_cancel = QPushButton(tr("Cancel")) # Should be Localized "Cancel"
        self.btn_cancel.clicked.connect(self.reject)
        self.button_layout.addWidget(self.btn_cancel)

        self.main_layout.addLayout(self.button_layout)

    def set_description(self, text):
        self.lbl_desc.setText(text)

    def get_data(self):
        """Override to return the generated configuration data."""
        return self.result_data
