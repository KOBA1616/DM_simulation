# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.editor.forms.card_form import CardEditForm
from dm_toolkit.gui.editor.forms.effect_form import EffectEditForm
from dm_toolkit.gui.editor.forms.action_form import ActionEditForm
from dm_toolkit.gui.editor.forms.spell_side_form import SpellSideForm
from dm_toolkit.gui.editor.forms.reaction_form import ReactionEditForm
from dm_toolkit.gui.editor.forms.command_form import CommandEditForm
from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm
from dm_toolkit.gui.editor.forms.keyword_form import KeywordEditForm
from dm_toolkit.gui.editor.forms.modifier_form import ModifierEditForm
from dm_toolkit.gui.editor.forms.option_form import OptionForm
from dm_toolkit.gui.localization import tr

class OptionEditPage(QWidget):
    structure_update_requested = pyqtSignal(str, dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr("Option selected. Add Actions to define behavior.")))
        btn = QPushButton(tr("Add Action"))
        btn.clicked.connect(lambda: self.structure_update_requested.emit("ADD_CHILD_ACTION", {}))
        layout.addWidget(btn)
        layout.addStretch()

class PropertyInspector(QWidget):
    # Forward signal from forms
    structure_update_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def _on_structure_update(self, command: str, data: dict):
        """Handle structure update requests from child forms."""
        # Intermediate processing (logging/validation) can be added here
        self.structure_update_requested.emit(command, data)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.label = QLabel(tr("Property Inspector"))
        layout.addWidget(self.label)

        self.stack = QStackedWidget()

        self.empty_page = QLabel(tr("Select an item to edit"))
        self.stack.addWidget(self.empty_page)

        # Replaced Label page with OptionForm
        self.option_form = OptionForm()
        self.stack.addWidget(self.option_form)

        self.cmd_branch_page = QLabel(tr("Branch selected. Add Actions to this branch."))
        self.cmd_branch_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.cmd_branch_page)

        self.card_form = CardEditForm()
        self.stack.addWidget(self.card_form)
        self.card_form.structure_update_requested.connect(self._on_structure_update)

        self.effect_form = EffectEditForm()
        self.stack.addWidget(self.effect_form)
        self.effect_form.structure_update_requested.connect(self._on_structure_update)

        # Unified Action UI replaces separate Action/Command editors
        self.unified_form = UnifiedActionForm()
        self.stack.addWidget(self.unified_form)
        self.unified_form.structure_update_requested.connect(self._on_structure_update)

        self.spell_side_form = SpellSideForm()
        self.stack.addWidget(self.spell_side_form)

        self.reaction_form = ReactionEditForm()
        self.stack.addWidget(self.reaction_form)

        # Keep old forms available for now but DO NOT add them to the visible stack
        # This effectively disables the legacy editors in the UI while preserving code.
        self.command_form = CommandEditForm()
        # Do not add to stack: self.stack.addWidget(self.command_form)
        # Do not connect signals to avoid UI interactions until removal/cleanup

        self.keyword_form = KeywordEditForm()
        self.stack.addWidget(self.keyword_form)
        self.keyword_form.structure_update_requested.connect(self._on_structure_update)

        self.modifier_form = ModifierEditForm()
        self.stack.addWidget(self.modifier_form)

        layout.addWidget(self.stack)

        # Initialize dispatch table
        self.form_map = {
            "CARD": self.card_form,
            "EFFECT": self.effect_form,
            # Use UnifiedActionForm for both ACTION and COMMAND types
            "ACTION": self.unified_form,
            "COMMAND": self.unified_form,
            "SPELL_SIDE": self.spell_side_form,
            "REACTION_ABILITY": self.reaction_form,
            "KEYWORDS": self.keyword_form,
            "MODIFIER": self.modifier_form,
            "OPTION": self.option_form, # Updated to use Form
            "CMD_BRANCH_TRUE": self.cmd_branch_page,
            "CMD_BRANCH_FALSE": self.cmd_branch_page,
        }

    def set_selection(self, index):
        if index is None or not index.isValid():
            self.stack.setCurrentWidget(self.empty_page)
            return

        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        item = index.model().itemFromIndex(index)

        widget = self.form_map.get(item_type, self.empty_page)

        if hasattr(widget, 'set_data'):
            widget.set_data(item)

        self.stack.setCurrentWidget(widget)
