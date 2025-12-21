# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QCheckBox, QLabel, QVBoxLayout
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.wizards.base_wizard import BaseWizardDialog

class HyperEnergyWizard(BaseWizardDialog):
    def __init__(self, parent=None):
        super().__init__(parent, tr("Hyper Energy Wizard"))
        self.set_description(tr("Enable Hyper Energy for this card. This mechanism allows you to tap creatures to reduce the summon cost."))
        self.setup_ui()

    def setup_ui(self):
        self.chk_enable = QCheckBox(tr("Enable Hyper Energy"))
        self.chk_enable.setChecked(True)
        self.content_layout.addWidget(self.chk_enable)

        # Info note
        lbl_info = QLabel(tr("Note: Ensure the card's Civilization is set correctly. The engine handles cost reduction automatically."))
        lbl_info.setStyleSheet("color: gray; font-style: italic;")
        lbl_info.setWordWrap(True)
        self.content_layout.addWidget(lbl_info)

    def get_data(self):
        return {
            "hyper_energy": self.chk_enable.isChecked()
        }
