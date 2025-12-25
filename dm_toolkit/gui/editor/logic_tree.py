# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QTreeView, QAbstractItemView, QMenu, QInputDialog
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import Qt, QModelIndex
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.data_manager import CardDataManager

class LogicTreeWidget(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.standard_model = QStandardItemModel()
        self.setModel(self.standard_model)
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
        self.data_manager = CardDataManager(self.standard_model)

        self.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if not indexes:
            return
        index = indexes[0]

        # Auto-expand if CARD is selected
        item = self.standard_model.itemFromIndex(index)
        if item:
            item_type = item.data(Qt.ItemDataRole.UserRole + 1)
            if item_type == "CARD":
                self._expand_card_tree(index)

    def _expand_card_tree(self, card_index):
        """Expands the card and its immediate children (Effects) to show Commands."""
        self.expand(card_index)
        item = self.standard_model.itemFromIndex(card_index)
        for i in range(item.rowCount()):
            child_index = item.child(i).index()
            self.expand(child_index)

    def mousePressEvent(self, event):
        # Default behavior: Click selects, Arrow click toggles expansion.
        # We removed the forced toggle on row click to prevent accidental collapsing
        # when trying to select an item.
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
             cmd_menu = menu.addMenu(tr("Add Command"))
             templates = self.data_manager.templates.get("commands", [])
             if not templates:
                 add_cmd_action = QAction(tr("Transition (Default)"), self)
                 add_cmd_action.triggered.connect(lambda: self.add_command_to_effect(index))
                 cmd_menu.addAction(add_cmd_action)
             else:
                 for tpl in templates:
                     action = QAction(tr(tpl['name']), self)
                     # Capture tpl['data'] in lambda default arg
                     action.triggered.connect(lambda checked, data=tpl['data']: self.add_command_to_effect(index, data))
                     cmd_menu.addAction(action)

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
            cmd_menu = menu.addMenu(tr("Add Command"))
            templates = self.data_manager.templates.get("commands", [])
            if not templates:
                add_cmd_action = QAction(tr("Transition (Default)"), self)
                add_cmd_action.triggered.connect(lambda: self.add_command_to_option(index))
                cmd_menu.addAction(add_cmd_action)
            else:
                for tpl in templates:
                    action = QAction(tr(tpl['name']), self)
                    action.triggered.connect(lambda checked, data=tpl['data']: self.add_command_to_option(index, data))
                    cmd_menu.addAction(action)

            remove_opt_action = QAction(tr("Remove Option"), self)
            remove_opt_action.triggered.connect(lambda: self.remove_current_item())
            menu.addAction(remove_opt_action)

        elif item_type == "COMMAND":
             remove_cmd = QAction(tr("Remove Command"), self)
             remove_cmd.triggered.connect(lambda: self.remove_current_item())
             menu.addAction(remove_cmd)

        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(pos))

    def replace_item_with_command(self, index, cmd_data):
        """Replaces a legacy Action item with a new Command item."""
        if not index.isValid(): return

        parent_item = self.standard_model.itemFromIndex(index.parent())
        row = index.row()

        # Remove old Action
        parent_item.removeRow(row)

        # Insert new Command at same position
        # Using insertRow with new item

        # We need to construct the command item hierarchy using data_manager
        # data_manager doesn't have public insert, so we do it via model or add helper

        # But wait, data_manager._create_command_item creates a QStandardItem
        cmd_item = self.data_manager._create_command_item(cmd_data)
        parent_item.insertRow(row, cmd_item)

        # Select the new item
        self.setCurrentIndex(cmd_item.index())

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
        parent_item = self.standard_model.itemFromIndex(parent_index)
        count = parent_item.rowCount() + 1

        new_item = QStandardItem(f"{tr('Option')} {count}")
        new_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
        new_item.setData({}, Qt.ItemDataRole.UserRole + 2)

        parent_item.appendRow(new_item)
        self.expand(parent_index)
        self.setCurrentIndex(new_item.index())

    def add_command_to_option(self, option_index, cmd_data=None):
        if not option_index.isValid(): return
        if cmd_data is None:
            cmd_data = {
                "type": "TRANSITION",
                "target_group": "NONE",
                "to_zone": "HAND",
                "target_filter": {}
            }

        # Deep copy to avoid reference issues
        import copy
        data_copy = copy.deepcopy(cmd_data)

        self.add_child_item(option_index, "COMMAND", data_copy, f"{tr('Command')}: {tr(data_copy.get('type', 'NONE'))}")

    def add_action_to_effect(self, effect_index, action_data=None):
        if not effect_index.isValid(): return
        if action_data is None:
             action_data = {"type": "MOVE_CARD", "from_zone": "DECK", "to_zone": "HAND"}
        self.add_child_item(effect_index, "ACTION", action_data, f"{tr('Action')}: {tr('MOVE_CARD')}")

    def add_action_to_option(self, option_index, action_data=None):
        if not option_index.isValid(): return
        if action_data is None:
             action_data = {"type": "MOVE_CARD", "from_zone": "DECK", "to_zone": "HAND"}
        self.add_child_item(option_index, "ACTION", action_data, f"{tr('Action')}: {tr('MOVE_CARD')}")

    def add_action_sibling(self, action_index, action_data=None):
        if not action_index.isValid(): return
        parent_index = action_index.parent()
        if not parent_index.isValid(): return

        if action_data is None:
             action_data = {"type": "MOVE_CARD", "from_zone": "DECK", "to_zone": "HAND"}

        self.add_child_item(parent_index, "ACTION", action_data, f"{tr('Action')}: {tr('MOVE_CARD')}")

    def add_command_to_effect(self, effect_index, cmd_data=None):
        if not effect_index.isValid(): return
        if cmd_data is None:
            cmd_data = {
                "type": "TRANSITION",
                "target_group": "NONE",
                "to_zone": "HAND",
                "target_filter": {}
            }

        # Deep copy
        import copy
        data_copy = copy.deepcopy(cmd_data)

        self.add_child_item(effect_index, "COMMAND", data_copy, f"{tr('Command')}: {tr(data_copy.get('type', 'NONE'))}")

    def add_action_to_effect(self, effect_index, action_data=None):
        if not effect_index.isValid(): return
        if action_data is None:
            action_data = {
                "type": "DRAW_CARD",
                "value": 1
            }

        # Deep copy
        import copy
        data_copy = copy.deepcopy(action_data)

        self.add_child_item(effect_index, "ACTION", data_copy, f"{tr('Action')}: {tr(data_copy.get('type', 'NONE'))}")

    def generate_branches_for_current(self):
        """Generates child branches for the currently selected command item."""
        index = self.currentIndex()
        if not index.isValid(): return

        item = self.standard_model.itemFromIndex(index)
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)

        if item_type == "COMMAND":
            self.data_manager.add_command_branches(item)
            self.expand(index)

    def add_effect_interactive(self, parent_index):
        if not parent_index.isValid(): return

        parent_item = self.standard_model.itemFromIndex(parent_index)
        role = parent_item.data(Qt.ItemDataRole.UserRole + 1)
        card_data = parent_item.data(Qt.ItemDataRole.UserRole + 2) or {}

        items = [tr("Triggered Ability"), tr("Static Ability")]

        # Check if we can add Reaction Ability
        # Only for CARD (not SPELL_SIDE) and type is not SPELL
        if role == "CARD":
             card_type = card_data.get('type', 'CREATURE')
             if card_type != "SPELL":
                  items.append(tr("Reaction Ability"))

        item, ok = QInputDialog.getItem(self, tr("Add Effect"), tr("Select Effect Type"), items, 0, False)

        if ok and item:
            if item == tr("Triggered Ability"):
                self.add_child_item(parent_index, "EFFECT",
                                    {"trigger": "ON_PLAY", "condition": {"type": "NONE"}, "actions": []},
                                    f"{tr('Effect')}: ON_PLAY")
            elif item == tr("Static Ability"):
                self.add_child_item(parent_index, "MODIFIER",
                                    {"type": "COST_MODIFIER", "value": -1, "condition": {"type": "NONE"}},
                                    f"{tr('Static')}: COST_MODIFIER")
            elif item == tr("Reaction Ability"):
                self.add_reaction(parent_index)

    def move_effect_item(self, item, target_type):
        """No-op as we use a flat structure now."""
        return

    def load_data(self, cards_data):
        # Save Expansion State
        expanded_ids = self._save_expansion_state()

        self.data_manager.load_data(cards_data)

        # Restore Expansion State
        self._restore_expansion_state(expanded_ids)

    def _save_expansion_state(self):
        """Saves the IDs of expanded items."""
        expanded_ids = set()
        root = self.standard_model.invisibleRootItem()
        self._traverse_save_expansion(root, expanded_ids)
        return expanded_ids

    def _traverse_save_expansion(self, item, expanded_ids):
        index = item.index()
        # Root index is invalid but we traverse its children
        if index.isValid() and self.isExpanded(index):
            # Use a unique identifier if possible, e.g., the name + type path?
            # Or rely on object structure. For cards, use ID.
            # For children, using a path-like string might be safer.
            # Here we use a path string: "0:0:1"
            path = self._get_item_path(item)
            expanded_ids.add(path)

        for i in range(item.rowCount()):
            self._traverse_save_expansion(item.child(i), expanded_ids)

    def _restore_expansion_state(self, expanded_ids):
        """Restores expansion state based on saved paths."""
        root = self.standard_model.invisibleRootItem()
        self._traverse_restore_expansion(root, expanded_ids)

    def _traverse_restore_expansion(self, item, expanded_ids):
        index = item.index()
        if index.isValid():
            path = self._get_item_path(item)
            if path in expanded_ids:
                self.setExpanded(index, True)

        for i in range(item.rowCount()):
            self._traverse_restore_expansion(item.child(i), expanded_ids)

    def _get_item_path(self, item):
        """Generates a simple path string 'row:row:row'."""
        path = []
        curr = item
        while curr:
            path.append(str(curr.row()))
            curr = curr.parent()
        return ":".join(reversed(path))

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
        self.standard_model.removeRow(idx.row(), idx.parent())

    def add_spell_side(self, card_index):
        if not card_index.isValid(): return
        card_item = self.standard_model.itemFromIndex(card_index)
        item = self.data_manager.add_spell_side_item(card_item)
        if item:
            self.setCurrentIndex(item.index())
            self.expand(card_index)
        return item

    def remove_spell_side(self, card_index):
        if not card_index.isValid(): return
        card_item = self.standard_model.itemFromIndex(card_index)
        self.data_manager.remove_spell_side_item(card_item)

    def add_rev_change(self, card_index):
        if not card_index.isValid(): return
        card_item = self.standard_model.itemFromIndex(card_index)
        eff_item = self.data_manager.add_revolution_change_logic(card_item)
        if eff_item:
            self.setCurrentIndex(eff_item.index())
            self.expand(card_index)
        return eff_item

    def remove_rev_change(self, card_index):
        if not card_index.isValid(): return
        card_item = self.standard_model.itemFromIndex(card_index)
        self.data_manager.remove_revolution_change_logic(card_item)
