# -*- coding: cp932 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.editor.forms.card_form import CardEditForm
from dm_toolkit.gui.editor.forms.effect_form import EffectEditForm
from dm_toolkit.gui.editor.forms.action_form import ActionEditForm
from dm_toolkit.gui.editor.forms.spell_side_form import SpellSideForm
from dm_toolkit.gui.editor.forms.reaction_form import ReactionEditForm
from dm_toolkit.gui.editor.forms.command_form import CommandEditForm
from dm_toolkit.gui.editor.forms.modifier_form import ModifierEditForm
from dm_toolkit.gui.localization import tr

class PropertyInspector(QWidget):
    # Forward signal from forms
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.label = QLabel(tr("Property Inspector"))
        layout.addWidget(self.label)

        self.stack = QStackedWidget()

        self.empty_page = QLabel(tr("Select an item to edit"))
        self.stack.addWidget(self.empty_page)

        self.option_page = QLabel(tr("Option selected. Use 'Add Action' to define behavior."))
        self.option_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.option_page)

        self.cmd_branch_page = QLabel(tr("Branch selected. Add Commands to this branch."))
        self.cmd_branch_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.cmd_branch_page)

        self.card_form = CardEditForm()
        self.stack.addWidget(self.card_form)
        self.card_form.structure_update_requested.connect(self.structure_update_requested.emit)

        self.effect_form = EffectEditForm()
        self.stack.addWidget(self.effect_form)
        self.effect_form.structure_update_requested.connect(self.structure_update_requested.emit)

        self.action_form = ActionEditForm()
        self.stack.addWidget(self.action_form)
        self.action_form.structure_update_requested.connect(self.structure_update_requested.emit)

        self.spell_side_form = SpellSideForm()
        self.stack.addWidget(self.spell_side_form)

        self.reaction_form = ReactionEditForm()
        self.stack.addWidget(self.reaction_form)

        self.command_form = CommandEditForm()
        self.stack.addWidget(self.command_form)

        self.modifier_form = ModifierEditForm()
        self.stack.addWidget(self.modifier_form)

        layout.addWidget(self.stack)

    def set_selection(self, index):
        if index is None or not index.isValid():
            self.stack.setCurrentWidget(self.empty_page)
            return

        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        item = index.model().itemFromIndex(index)

        if item_type == "CARD":
            self.card_form.set_data(item)
            self.stack.setCurrentWidget(self.card_form)
        elif item_type == "EFFECT":
            self.effect_form.set_data(item)
            self.stack.setCurrentWidget(self.effect_form)
        elif item_type == "ACTION":
            self.action_form.set_data(item)
            self.stack.setCurrentWidget(self.action_form)
        elif item_type == "COMMAND":
            self.command_form.set_data(item)
            self.stack.setCurrentWidget(self.command_form)
        elif item_type == "SPELL_SIDE":
            self.spell_side_form.set_data(item)
            self.stack.setCurrentWidget(self.spell_side_form)
        elif item_type == "REACTION_ABILITY":
            self.reaction_form.set_data(item)
            self.stack.setCurrentWidget(self.reaction_form)
        elif item_type == "MODIFIER":
            self.modifier_form.set_data(item)
            self.stack.setCurrentWidget(self.modifier_form)
        elif item_type == "OPTION":
            self.stack.setCurrentWidget(self.option_page)
        elif item_type == "CMD_BRANCH_TRUE" or item_type == "CMD_BRANCH_FALSE":
            self.stack.setCurrentWidget(self.cmd_branch_page)
        else:
            self.stack.setCurrentWidget(self.empty_page)
