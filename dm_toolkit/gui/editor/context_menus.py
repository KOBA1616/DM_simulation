# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction
from dm_toolkit.gui.i18n import tr

class LogicTreeContextMenuHandler:
    def __init__(self, tree_widget):
        self.tree_widget = tree_widget

    def show_context_menu(self, pos):
        index = self.tree_widget.indexAt(pos)
        if not index.isValid():
            return

        item_type = self.tree_widget.data_manager.get_item_type(index)
        menu = QMenu(self.tree_widget)

        if item_type == "CARD" or item_type == "SPELL_SIDE":
            self._build_card_context_menu(menu, index, item_type)
        elif item_type == "EFFECT":
            self._build_effect_context_menu(menu, index)
        elif item_type in ["MODIFIER", "REACTION_ABILITY"]:
            self._add_remove_action(menu)
        elif item_type == "OPTION":
            self._build_option_context_menu(menu, index)
        elif item_type == "COMMAND":
            self._add_remove_action(menu)
        elif item_type in ["CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
            add_cmd_br = QAction(tr("Add Command"), self.tree_widget)
            add_cmd_br.triggered.connect(lambda: self.tree_widget._add_command_to_branch(index))
            menu.addAction(add_cmd_br)

        if not menu.isEmpty():
            vp = self.tree_widget.viewport()
            if vp is not None:
                menu.exec(vp.mapToGlobal(pos))

    def _build_card_context_menu(self, menu, index, item_type):
        add_eff_action = QAction(tr("Add Effect"), self.tree_widget)
        add_eff_action.triggered.connect(lambda: self.tree_widget.add_effect_interactive(index))
        menu.addAction(add_eff_action)

        add_static_action = QAction(tr("Add Static Ability"), self.tree_widget)
        add_static_action.triggered.connect(lambda: self.tree_widget.add_static(index))
        menu.addAction(add_static_action)

        if item_type != "SPELL_SIDE":
            add_reaction_action = QAction(tr("Add Reaction Ability"), self.tree_widget)
            add_reaction_action.triggered.connect(lambda: self.tree_widget.add_reaction(index))
            menu.addAction(add_reaction_action)

    def _build_effect_context_menu(self, menu, index):
        self._add_command_submenu(menu, index, self.tree_widget.add_command_to_effect)
        menu.addSeparator()
        self._add_remove_action(menu)

    def _build_option_context_menu(self, menu, index):
        self._add_command_submenu(menu, index, self.tree_widget.add_command_to_option)
        menu.addSeparator()
        self._add_remove_action(menu, label=tr("Remove Option"))

    def _add_command_submenu(self, menu, index, callback):
        cmd_menu = menu.addMenu(tr("Add Command"))
        # Fix access to templates via template_manager
        templates = self.tree_widget.data_manager.template_manager.templates.get("commands", [])

        # Default Transition
        add_cmd_action = QAction(tr("Transition (Default)"), self.tree_widget)
        add_cmd_action.triggered.connect(lambda checked: callback(index))
        cmd_menu.addAction(add_cmd_action)
        cmd_menu.addSeparator()

        if not templates:
            warning = QAction(tr("(No Templates Found)"), self.tree_widget)
            warning.setEnabled(False)
            cmd_menu.addAction(warning)
        else:
            for tpl in templates:
                action = QAction(tr(tpl['name']), self.tree_widget)
                # Use default argument binding properly
                action.triggered.connect(lambda checked, d=tpl['data']: callback(index, d))
                cmd_menu.addAction(action)

    def _add_remove_action(self, menu, label=None):
        label = label or tr("Remove Item")
        remove_action = QAction(label, self.tree_widget)
        remove_action.triggered.connect(lambda: self.tree_widget.remove_current_item())
        menu.addAction(remove_action)
