# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QTreeView, QAbstractItemView, QMenu, QInputDialog
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import Qt, QModelIndex
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.data_manager import CardDataManager

class LogicTreeWidget(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = QStandardItemModel()
        self.setModel(self.model)
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # Initialize Data Manager
        self.data_manager = CardDataManager(self.model)

        self.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if not indexes:
            return
        index = indexes[0]
        # Could trigger property inspector update here if needed,
        # but usually Main Window handles checking selection.

    def mousePressEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid() and event.button() == Qt.MouseButton.LeftButton:
             item_type = index.data(Qt.ItemDataRole.UserRole + 1)
             expandable_types = ["CARD", "SPELL_SIDE", "EFFECT", "ACTION", "OPTION", "COMMAND", "CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]
             if item_type in expandable_types:
                 # Toggle expansion on row click
                 if self.isExpanded(index):
                     self.collapse(index)
                 else:
                     self.expand(index)
        super().mousePressEvent(event)

    def show_context_menu(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return

        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        menu = QMenu(self)

        # Logic Mask: Get Card Type to filter options
        card_type = "CREATURE" # Default
        # Traverse up to find CARD
        temp = index
        while temp.isValid():
            if temp.data(Qt.ItemDataRole.UserRole + 1) == "CARD":
                cdata = temp.data(Qt.ItemDataRole.UserRole + 2)
                card_type = cdata.get('type', 'CREATURE')
                break
            temp = temp.parent()

        is_spell = (card_type == "SPELL")

        if item_type == "CARD" or item_type == "SPELL_SIDE":
            add_eff_action = QAction(tr("Add Effect"), self)
            add_eff_action.triggered.connect(lambda: self.add_effect_interactive(index))
            menu.addAction(add_eff_action)

            if not is_spell and item_type != "SPELL_SIDE":
                 add_reaction_action = QAction(tr("Add Reaction Ability"), self)
                 add_reaction_action.triggered.connect(lambda: self.add_reaction(index))
                 menu.addAction(add_reaction_action)

        elif item_type == "EFFECT":
             add_act_action = QAction(tr("Add Action"), self)
             add_act_action.triggered.connect(lambda: self.add_action_to_effect(index))
             menu.addAction(add_act_action)

             remove_action = QAction(tr("Remove Item"), self)
             remove_action.triggered.connect(lambda: self.remove_current_item())
             menu.addAction(remove_action)

        elif item_type == "MODIFIER" or item_type == "REACTION_ABILITY":
             remove_action = QAction(tr("Remove Item"), self)
             remove_action.triggered.connect(lambda: self.remove_current_item())
             menu.addAction(remove_action)

        elif item_type == "ACTION":
            act_data = index.data(Qt.ItemDataRole.UserRole + 2)
            if act_data.get('type') == "SELECT_OPTION":
                add_opt_action = QAction(tr("Add Option"), self)
                add_opt_action.triggered.connect(lambda: self.add_option(index))
                menu.addAction(add_opt_action)

            remove_action = QAction(tr("Remove Action"), self)
            remove_action.triggered.connect(lambda: self.remove_current_item())
            menu.addAction(remove_action)

        elif item_type == "OPTION":
            add_act_action = QAction(tr("Add Action"), self)
            add_act_action.triggered.connect(lambda: self.add_action_to_option(index))
            menu.addAction(add_act_action)

            remove_opt_action = QAction(tr("Remove Option"), self)
            remove_opt_action.triggered.connect(lambda: self.remove_current_item())
            menu.addAction(remove_opt_action)

        elif item_type == "COMMAND":
             remove_cmd = QAction(tr("Remove Command"), self)
             remove_cmd.triggered.connect(lambda: self.remove_current_item())
             menu.addAction(remove_cmd)

        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(pos))

    def add_keywords(self, parent_index):
        if not parent_index.isValid(): return
        self.add_child_item(parent_index, "KEYWORDS", {}, tr("Keywords"))

    def add_trigger(self, parent_index):
        if not parent_index.isValid(): return
        # Default Trigger Data
        eff_data = {
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "actions": []
        }
        self.add_child_item(parent_index, "EFFECT", eff_data, f"{tr('Effect')}: ON_PLAY")

    def add_static(self, parent_index):
        if not parent_index.isValid(): return
        # Default Static Data
        mod_data = {
            "type": "COST_MODIFIER",
            "value": -1,
            "condition": {"type": "NONE"}
        }
        self.add_child_item(parent_index, "MODIFIER", mod_data, f"{tr('Static')}: COST_MODIFIER")

    def add_reaction(self, parent_index):
        if not parent_index.isValid(): return
        # Default Reaction Data
        ra_data = {
            "type": "NINJA_STRIKE",
            "cost": 4,
            "zone": "HAND",
            "condition": {
                "trigger_event": "ON_BLOCK_OR_ATTACK",
                "civilization_match": True,
                "mana_count_min": 0
            }
        }
        self.add_child_item(parent_index, "REACTION_ABILITY", ra_data, f"{tr('Reaction Ability')}: NINJA_STRIKE")

    def add_option(self, parent_index):
        if not parent_index.isValid(): return
        parent_item = self.model.itemFromIndex(parent_index)
        count = parent_item.rowCount() + 1

        new_item = QStandardItem(f"{tr('Option')} {count}")
        new_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
        new_item.setData({}, Qt.ItemDataRole.UserRole + 2)

        parent_item.appendRow(new_item)
        self.expand(parent_index)
        self.setCurrentIndex(new_item.index())

    def add_action_to_option(self, option_index):
        if not option_index.isValid(): return
        act_data = {"type": "NONE"}
        self.add_child_item(option_index, "ACTION", act_data, f"{tr('Action')}: NONE")

    def add_action_to_effect(self, effect_index):
        if not effect_index.isValid(): return
        act_data = {"type": "DESTROY", "scope": "TARGET_SELECT", "value1": 0, "filter": {}}
        self.add_child_item(effect_index, "ACTION", act_data, f"{tr('Action')}: {tr('DESTROY')}")

    def add_effect_interactive(self, parent_index):
        if not parent_index.isValid(): return

        items = [tr("Triggered Ability"), tr("Static Ability")]
        item, ok = QInputDialog.getItem(self, tr("Add Effect"), tr("Select Effect Type"), items, 0, False)

        if ok and item:
            if item == tr("Triggered Ability"):
                self.add_child_item(parent_index, "EFFECT",
                                    {"trigger": "ON_PLAY", "condition": {"type": "NONE"}, "actions": []},
                                    f"{tr('Effect')}: ON_PLAY")
            else:
                self.add_child_item(parent_index, "MODIFIER",
                                    {"type": "COST_MODIFIER", "value": -1, "condition": {"type": "NONE"}},
                                    f"{tr('Static')}: COST_MODIFIER")

    def move_effect_item(self, item, target_type):
        """No-op as we use a flat structure now."""
        return

    def load_data(self, cards_data):
        self.data_manager.load_data(cards_data)

    def get_full_data_from_model(self):
        return self.data_manager.get_full_data()

    def add_new_card(self):
        item = self.data_manager.add_new_card()
        if item:
            self.setCurrentIndex(item.index())
            self.expand(item.index())
        return item

    def add_child_item(self, parent_index, item_type, data, label):
        new_item = self.data_manager.add_child_item(parent_index, item_type, data, label)
        if new_item:
            self.setExpanded(parent_index, True)
            self.setCurrentIndex(new_item.index())
        return new_item

    def remove_current_item(self):
        idx = self.currentIndex()
        if not idx.isValid(): return
        self.model.removeRow(idx.row(), idx.parent())

    def add_spell_side(self, card_index):
        if not card_index.isValid(): return
        card_item = self.model.itemFromIndex(card_index)
        item = self.data_manager.add_spell_side_item(card_item)
        if item:
            self.setCurrentIndex(item.index())
            self.expand(card_index)
        return item

    def remove_spell_side(self, card_index):
        if not card_index.isValid(): return
        card_item = self.model.itemFromIndex(card_index)
        self.data_manager.remove_spell_side_item(card_item)

    def add_rev_change(self, card_index):
        if not card_index.isValid(): return
        card_item = self.model.itemFromIndex(card_index)
        eff_item = self.data_manager.add_revolution_change_logic(card_item)
        if eff_item:
            self.setCurrentIndex(eff_item.index())
            self.expand(card_index)
        return eff_item

    def remove_rev_change(self, card_index):
        if not card_index.isValid(): return
        card_item = self.model.itemFromIndex(card_index)
        self.data_manager.remove_revolution_change_logic(card_item)
