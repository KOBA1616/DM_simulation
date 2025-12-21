# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QFormLayout, QComboBox, QSpinBox, QLineEdit
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.wizards.base_wizard import BaseWizardDialog
from dm_toolkit.gui.editor.forms.parts.civilization_widget import CivilizationSelector

class RevolutionChangeWizard(BaseWizardDialog):
    def __init__(self, parent=None):
        super().__init__(parent, tr("Revolution Change Wizard"))
        self.set_description(tr("Configure the conditions for Revolution Change. This will automatically generate the 'Revolution Change' keyword and the necessary trigger logic."))
        self.setup_ui()

    def setup_ui(self):
        form_layout = QFormLayout()
        self.content_layout.addLayout(form_layout)

        # Civilization
        self.civ_selector = CivilizationSelector()
        form_layout.addRow(tr("Civilization"), self.civ_selector)

        # Race
        self.race_edit = QLineEdit()
        self.race_edit.setPlaceholderText(tr("e.g. Dragon"))
        form_layout.addRow(tr("Race"), self.race_edit)

        # Cost
        self.cost_spin = QSpinBox()
        self.cost_spin.setRange(0, 99)
        self.cost_spin.setValue(5)
        form_layout.addRow(tr("Minimum Cost"), self.cost_spin)

    def get_data(self):
        """
        Returns a dictionary structure compatible with CardDataManager.add_revolution_change_logic
        or the raw JSON structure needed.
        """
        civs = self.civ_selector.get_selected_civs()
        race = self.race_edit.text().strip()
        min_cost = self.cost_spin.value()

        # Construct the FilterDef
        filter_def = {
            "civilizations": civs,
            "min_cost": min_cost
        }
        if race:
            filter_def["races"] = [race]

        return filter_def
