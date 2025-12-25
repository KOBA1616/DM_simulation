# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QTreeView, QAbstractItemView, QMenu, QInputDialog, QMessageBox
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import Qt, QModelIndex
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.data_manager import CardDataManager
import uuid

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
        """Recursively expands the card and all its descendants."""
        self.expand(card_index)
        self._expand_children_recursive(card_index)

    def _expand_children_recursive(self, parent_index):
        item = self.standard_model.itemFromIndex(parent_index)
        for i in range(item.rowCount()):
            child_index = item.child(i).index()
            self.expand(child_index)
            self._expand_children_recursive(child_index)

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
                # Null Safety Fix: Default to empty dict if data is missing
                cdata = temp.data(Qt.ItemDataRole.UserRole + 2) or {}
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

            # Bulk Conversion Action
            convert_all_action = QAction(tr("Convert All Legacy Actions to Commands"), self)
            convert_all_action.triggered.connect(lambda: self.convert_all_legacy_actions_in_node(index))
            menu.addAction(convert_all_action)

        elif item_type == "EFFECT":
             cmd_menu = menu.addMenu(tr("Add Command"))
             templates = self.data_manager.templates.get("commands", [])

             if not templates:
                 warning = QAction(tr("(No Templates Found)"), self)
                 warning.setEnabled(False)
                 cmd_menu.addAction(warning)

                 add_cmd_action = QAction(tr("Transition (Default)"), self)
                 add_cmd_action.triggered.connect(lambda checked: self.add_command_to_effect(index))
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
            act_data = index.data(Qt.ItemDataRole.UserRole + 2) or {}
            if act_data.get('type') == "SELECT_OPTION":
                add_opt_action = QAction(tr("Add Option"), self)
                add_opt_action.triggered.connect(lambda: self.add_option(index))
                menu.addAction(add_opt_action)

            replace_cmd_action = QAction(tr("Convert to Command"), self)
            replace_cmd_action.triggered.connect(lambda: self.replace_item_with_command(index, self._convert_action_tree_to_command(self.standard_model.itemFromIndex(index))))
            menu.addAction(replace_cmd_action)

            remove_action = QAction(tr("Remove Action"), self)
            remove_action.triggered.connect(lambda: self.remove_current_item())
            menu.addAction(remove_action)

        elif item_type == "OPTION":
            cmd_menu = menu.addMenu(tr("Add Command"))
            templates = self.data_manager.templates.get("commands", [])

            if not templates:
                warning = QAction(tr("(No Templates Found)"), self)
                warning.setEnabled(False)
                cmd_menu.addAction(warning)

                add_cmd_action = QAction(tr("Transition (Default)"), self)
                add_cmd_action.triggered.connect(lambda checked: self.add_command_to_option(index))
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

    def convert_all_legacy_actions_in_node(self, index):
        """Recursively converts all Actions to Commands starting from the given node."""
        if not index.isValid(): return

        item = self.standard_model.itemFromIndex(index)
        count, warnings = self._recursive_convert_actions(item)
        if count > 0:
            msg = f"{tr('Converted')} {count} {tr('actions to commands.')}"
            if warnings > 0:
                msg += f"\n\n{tr('Warnings')}: {warnings} {tr('items require attention.')}\n"
                msg += tr("(Look for items marked with 'Legacy Warning')")
                QMessageBox.warning(self, tr("Conversion Complete"), msg)
            else:
                QMessageBox.information(self, tr("Conversion Complete"), msg)
        else:
            QMessageBox.information(self, tr("Conversion Info"), tr("No legacy actions found to convert."))

    def _recursive_convert_actions(self, item):
        """
        Traverses the tree and converts ACTION items to COMMAND items.
        Returns (converted_count, warning_count).
        """
        converted_count = 0
        warning_count = 0
        # Iterate backwards to safely remove/insert rows
        for i in reversed(range(item.rowCount())):
            child = item.child(i)
            child_type = child.data(Qt.ItemDataRole.UserRole + 1)

            if child_type == "ACTION":
                # Convert this Action (and its children)
                cmd_data = self._convert_action_tree_to_command(child)

                # Check for warnings in this conversion branch
                w = self._scan_warnings_in_cmd(cmd_data)
                warning_count += w

                self.replace_item_with_command(child.index(), cmd_data)
                converted_count += 1
                # Note: replace_item_with_command reconstructs the node, so we don't recurse into the old child.
                # However, _convert_action_tree_to_command ALREADY handled the recursion for options.
            else:
                # Recurse deeper
                c, w = self._recursive_convert_actions(child)
                converted_count += c
                warning_count += w
        return converted_count, warning_count

    def _scan_warnings_in_cmd(self, cmd_data):
        w = 0
        if cmd_data.get('legacy_warning', False):
            w += 1

        if 'options' in cmd_data:
            for opt_list in cmd_data['options']:
                for sub_cmd in opt_list:
                    w += self._scan_warnings_in_cmd(sub_cmd)
        return w

    def replace_item_with_command(self, index, cmd_data):
        """Replaces a legacy Action item with a new Command item."""
        if not index.isValid(): return

        parent_item = self.standard_model.itemFromIndex(index.parent())
        old_item = self.standard_model.itemFromIndex(index)
        row = index.row()

        # 1. Capture old structure if it has children (OPTIONS)
        preserved_options_data = []
        if old_item.rowCount() > 0:
             # Check if children are OPTIONS
             for i in range(old_item.rowCount()):
                  child = old_item.child(i)
                  if child.data(Qt.ItemDataRole.UserRole + 1) == "OPTION":
                      # Collect actions inside
                      opt_actions_data = []
                      for k in range(child.rowCount()):
                          act_child = child.child(k)
                          # We only care about ACTIONs to convert
                          if act_child.data(Qt.ItemDataRole.UserRole + 1) == "ACTION":
                               # Recursively capture and convert
                               c_cmd = self._convert_action_tree_to_command(act_child)
                               opt_actions_data.append(c_cmd)
                          elif act_child.data(Qt.ItemDataRole.UserRole + 1) == "COMMAND":
                               # Already a command, preserve it
                               opt_actions_data.append(act_child.data(Qt.ItemDataRole.UserRole + 2))
                      preserved_options_data.append(opt_actions_data)

        # 2. Inject options into cmd_data if exists
        # NOTE: _convert_action_tree_to_command already handles options conversion!
        # If replace_item_with_command is called with the result of _convert..., cmd_data has 'options'.
        # But if we collected existing COMMAND nodes (mixed model), we need to ensure structure matches.

        # If cmd_data ALREADY has options (from conversion), we prefer that.
        # But wait, mixed models might have COMMANDs inside ACTION options?
        # _convert_action_tree_to_command recurses. If it finds COMMAND inside ACTION, it needs to handle it.
        # Let's fix _convert_action_tree_to_command to handle mixed children.

        # 3. Remove old Action
        if parent_item is None:
            # If parent is None, it might be the root item (top-level item)
            # index.parent() is invalid for top-level items.
            if not index.parent().isValid():
                parent_item = self.standard_model.invisibleRootItem()

        if parent_item is None:
             return

        parent_item.removeRow(row)

        # 4. Insert new Command at same position
        cmd_item = self.data_manager.create_command_item(cmd_data)
        parent_item.insertRow(row, cmd_item)

        # Select the new item
        self.setCurrentIndex(cmd_item.index())
        self.expand(cmd_item.index()) # Expand to show preserved children

    def _convert_action_tree_to_command(self, action_item):
        """Recursively converts an Action Item and its children to Command Data."""
        from dm_toolkit.gui.editor.action_converter import ActionConverter

        act_data = action_item.data(Qt.ItemDataRole.UserRole + 2)
        cmd_data = ActionConverter.convert(act_data)

        # Check for children (Options)
        options_list = []
        if action_item.rowCount() > 0:
            for i in range(action_item.rowCount()):
                child = action_item.child(i)
                if child.data(Qt.ItemDataRole.UserRole + 1) == "OPTION":
                    opt_cmds = []
                    for k in range(child.rowCount()):
                        sub_item = child.child(k)
                        if sub_item.data(Qt.ItemDataRole.UserRole + 1) == "ACTION":
                            opt_cmds.append(self._convert_action_tree_to_command(sub_item))
                        elif sub_item.data(Qt.ItemDataRole.UserRole + 1) == "COMMAND":
                            # Preserve existing commands
                            opt_cmds.append(sub_item.data(Qt.ItemDataRole.UserRole + 2))
                    options_list.append(opt_cmds)

        if options_list:
            cmd_data['options'] = options_list

        return cmd_data

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
        self.add_child_item(parent_index, "EFFECT", eff_data, f"{tr('Effect')}: {tr('ON_PLAY')}")

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
        new_item.setData({'uid': str(uuid.uuid4())}, Qt.ItemDataRole.UserRole + 2)

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
        if 'uid' in data_copy: del data_copy['uid']

        label = self.data_manager.format_command_label(data_copy)
        self.add_child_item(option_index, "COMMAND", data_copy, label)

    def add_action_sibling(self, action_index, action_data=None):
        if not action_index.isValid(): return
        parent_index = action_index.parent()
        if not parent_index.isValid(): return

        if action_data is None:
             action_data = {"type": "SELECT_TARGET", "filter": {"zones": ["BATTLE_ZONE"], "count": 1}}

        import copy
        data_copy = copy.deepcopy(action_data)
        if 'uid' in data_copy: del data_copy['uid']

        label = self.data_manager.format_action_label(data_copy)
        self.add_child_item(parent_index, "ACTION", data_copy, label)

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
        if 'uid' in data_copy: del data_copy['uid']

        label = self.data_manager.format_command_label(data_copy)
        self.add_child_item(effect_index, "COMMAND", data_copy, label)

    def add_action_to_effect(self, effect_index, action_data=None):
        if not effect_index.isValid(): return
        if action_data is None:
             action_data = {"type": "SELECT_TARGET", "filter": {"zones": ["BATTLE_ZONE"], "count": 1}}

        import copy
        data_copy = copy.deepcopy(action_data)
        if 'uid' in data_copy: del data_copy['uid']

        label = self.data_manager.format_action_label(data_copy)
        self.add_child_item(effect_index, "ACTION", data_copy, label)

    def add_action_to_option(self, option_index, action_data=None):
        if not option_index.isValid(): return
        if action_data is None:
             action_data = {"type": "SELECT_TARGET", "filter": {"zones": ["BATTLE_ZONE"], "count": 1}}

        import copy
        data_copy = copy.deepcopy(action_data)
        if 'uid' in data_copy: del data_copy['uid']

        label = self.data_manager.format_action_label(data_copy)
        self.add_child_item(option_index, "ACTION", data_copy, label)

    def add_command_contextual(self, cmd_data=None):
        idx = self.currentIndex()
        if not idx.isValid(): return

        item = self.standard_model.itemFromIndex(idx)
        type_ = item.data(Qt.ItemDataRole.UserRole + 1)

        if type_ == "EFFECT":
            self.add_command_to_effect(idx, cmd_data)
        elif type_ == "OPTION":
            self.add_command_to_option(idx, cmd_data)
        elif type_ in ["COMMAND", "ACTION"]:
             # Sibling or Branch
             parent = item.parent()
             if parent:
                 parent_type = parent.data(Qt.ItemDataRole.UserRole + 1)
                 if parent_type == "EFFECT":
                     self.add_command_to_effect(parent.index(), cmd_data)
                 elif parent_type == "OPTION":
                     self.add_command_to_option(parent.index(), cmd_data)
                 elif parent_type in ["CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
                     self._add_command_to_branch(parent.index(), cmd_data)
        elif type_ in ["CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
             self._add_command_to_branch(idx, cmd_data)

    def add_action_contextual(self, action_data=None):
        idx = self.currentIndex()
        if not idx.isValid(): return

        item = self.standard_model.itemFromIndex(idx)
        type_ = item.data(Qt.ItemDataRole.UserRole + 1)

        if type_ == "EFFECT":
            self.add_action_to_effect(idx, action_data)
        elif type_ == "OPTION":
            self.add_action_to_option(idx, action_data)
        elif type_ in ["COMMAND", "ACTION"]:
             # Sibling
             parent = item.parent()
             if parent:
                 parent_type = parent.data(Qt.ItemDataRole.UserRole + 1)
                 if parent_type == "EFFECT":
                     self.add_action_to_effect(parent.index(), action_data)
                 elif parent_type == "OPTION":
                     self.add_action_to_option(parent.index(), action_data)

    def _add_command_to_branch(self, branch_index, cmd_data=None):
        if not branch_index.isValid(): return
        if cmd_data is None:
            cmd_data = {
                "type": "TRANSITION",
                "target_group": "NONE",
                "to_zone": "HAND",
                "target_filter": {}
            }

        import copy
        data_copy = copy.deepcopy(cmd_data)
        if 'uid' in data_copy: del data_copy['uid']
        self.add_child_item(branch_index, "COMMAND", data_copy, f"{tr('Command')}: {tr(data_copy.get('type', 'NONE'))}")

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
        """Updates the item's visual state (Label) to match the new type."""
        print(f"DEBUG: move_effect_item called with target {target_type}")
        data = item.data(Qt.ItemDataRole.UserRole + 2) or {}
        print(f"DEBUG: item data = {data}")

        if target_type == "TRIGGERED":
            item.setData("EFFECT", Qt.ItemDataRole.UserRole + 1)
            trigger = data.get('trigger', 'NONE')
            item.setText(f"{tr('Effect')}: {tr(trigger)}")

        elif target_type == "STATIC":
            item.setData("MODIFIER", Qt.ItemDataRole.UserRole + 1)
            mtype = data.get('type', data.get('layer_type', 'NONE'))
            print(f"DEBUG: Setting text to Static: {mtype}")
            item.setText(f"{tr('Static')}: {tr(mtype)}")

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
        """Generates a path string using UIDs if available, else row indices."""
        path = []
        curr = item
        # Stop if we hit the invisible root item to avoid including it in the path
        root = self.standard_model.invisibleRootItem()
        while curr and curr != root:
            data = curr.data(Qt.ItemDataRole.UserRole + 2)
            if data and isinstance(data, dict) and 'uid' in data:
                path.append(f"uid_{data['uid']}")
            else:
                path.append(f"row_{curr.row()}")
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

        row = idx.row()
        parent = idx.parent()

        self.standard_model.removeRow(row, parent)

        # Try to select neighbor to prevent stale UI
        if self.standard_model.rowCount(parent) > row:
            # Select next item which is now at 'row'
            new_idx = self.standard_model.index(row, 0, parent)
            self.setCurrentIndex(new_idx)
        elif self.standard_model.rowCount(parent) > 0:
            # Select last item
            new_idx = self.standard_model.index(self.standard_model.rowCount(parent) - 1, 0, parent)
            self.setCurrentIndex(new_idx)
        else:
            # Select parent
            if parent.isValid():
                self.setCurrentIndex(parent)

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
