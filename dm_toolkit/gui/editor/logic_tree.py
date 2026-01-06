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

        sel = self.selectionModel()
        if sel is not None:
            sel.selectionChanged.connect(self.on_selection_changed)

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if not indexes:
            return
        index = indexes[0]

        # Auto-expand if CARD is selected
        item_type = self.data_manager.get_item_type(index)
        if item_type == "CARD":
            self._expand_card_tree(index)

    def _expand_card_tree(self, card_index):
        """Recursively expands the card and all its descendants."""
        self.expand(card_index)
        self._expand_children_recursive(card_index)

    def _expand_children_recursive(self, parent_index):
        item = self.standard_model.itemFromIndex(parent_index)
        if item is None:
            return
        for i in range(item.rowCount()):
            child = item.child(i)
            if child is None:
                continue
            child_index = child.index()
            self.expand(child_index)
            self._expand_children_recursive(child_index)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def show_context_menu(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return

        item_type = self.data_manager.get_item_type(index)
        menu = QMenu(self)

        # Logic Mask: Get Card Type to filter options
        card_type = self.data_manager.get_card_context_type(index)
        is_spell = (card_type == "SPELL")

        if item_type == "CARD" or item_type == "SPELL_SIDE":
            add_eff_action = QAction(tr("Add Effect"), self)
            add_eff_action.triggered.connect(lambda: self.add_effect_interactive(index))
            menu.addAction(add_eff_action)

            # Removed restriction on Spells, as some (e.g. Strike Back) can apply to Spells.
            if item_type != "SPELL_SIDE":
                 add_reaction_action = QAction(tr("Add Reaction Ability"), self)
                 add_reaction_action.triggered.connect(lambda: self.add_reaction(index))
                 menu.addAction(add_reaction_action)

        elif item_type == "EFFECT":
             cmd_menu = menu.addMenu(tr("Add Command"))
             templates = self.data_manager.templates.get("commands", [])

             # Always add Default Transition option
             add_cmd_action = QAction(tr("Transition (Default)"), self)
             add_cmd_action.triggered.connect(lambda checked: self.add_command_to_effect(index))
             if cmd_menu is not None:
                 cmd_menu.addAction(add_cmd_action)

             if cmd_menu is not None:
                 cmd_menu.addSeparator()

             if not templates:
                 warning = QAction(tr("(No Templates Found)"), self)
                 warning.setEnabled(False)
                 if cmd_menu is not None:
                     cmd_menu.addAction(warning)
             else:
                 for tpl in templates:
                     action = QAction(tr(tpl['name']), self)
                     action.triggered.connect(lambda checked, data=tpl['data']: self.add_command_to_effect(index, data))
                     if cmd_menu is not None:
                         cmd_menu.addAction(action)

             remove_action = QAction(tr("Remove Item"), self)
             remove_action.triggered.connect(lambda: self.remove_current_item())
             menu.addAction(remove_action)

        elif item_type == "MODIFIER" or item_type == "REACTION_ABILITY":
             remove_action = QAction(tr("Remove Item"), self)
             remove_action.triggered.connect(lambda: self.remove_current_item())
             menu.addAction(remove_action)

        elif item_type == "OPTION":
            cmd_menu = menu.addMenu(tr("Add Command"))
            templates = self.data_manager.templates.get("commands", [])

            # Always add Default Transition option
            add_cmd_action = QAction(tr("Transition (Default)"), self)
            add_cmd_action.triggered.connect(lambda checked: self.add_command_to_option(index))
            if cmd_menu is not None:
                cmd_menu.addAction(add_cmd_action)

            if cmd_menu is not None:
                cmd_menu.addSeparator()

            if not templates:
                warning = QAction(tr("(No Templates Found)"), self)
                warning.setEnabled(False)
                if cmd_menu is not None:
                    cmd_menu.addAction(warning)
            else:
                for tpl in templates:
                    action = QAction(tr(tpl['name']), self)
                    action.triggered.connect(lambda checked, data=tpl['data']: self.add_command_to_option(index, data))
                    if cmd_menu is not None:
                        cmd_menu.addAction(action)

            remove_opt_action = QAction(tr("Remove Option"), self)
            remove_opt_action.triggered.connect(lambda: self.remove_current_item())
            menu.addAction(remove_opt_action)

        elif item_type == "COMMAND":
             remove_cmd = QAction(tr("Remove Command"), self)
             remove_cmd.triggered.connect(lambda: self.remove_current_item())
             menu.addAction(remove_cmd)

        # Add common helpers for CMD branches
        if item_type in ["CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
             add_cmd_br = QAction(tr("Add Command"), self)
             add_cmd_br.triggered.connect(lambda: self._add_command_to_branch(index))
             menu.addAction(add_cmd_br)

        if not menu.isEmpty():
            vp = self.viewport()
            if vp is not None:
                menu.exec(vp.mapToGlobal(pos))

    def convert_all_legacy_actions_in_node(self, index):
        """Preview then (optionally) convert all Actions to Commands starting from the given node."""
        if not index.isValid(): return

        item = self.standard_model.itemFromIndex(index)
        # Delegate collection to data manager
        preview_items = self.data_manager.collect_conversion_preview(item)

        if not preview_items:
            QMessageBox.information(self, tr("Conversion Info"), tr("No legacy actions found to convert."))
            return

        # Show preview dialog and apply only if user accepts
        from dm_toolkit.gui.editor.forms.convert_batch_preview_dialog import ConvertBatchPreviewDialog
        dlg = ConvertBatchPreviewDialog(self, preview_items)
        res = dlg.exec()
        if res == dlg.Accepted:
            # Delegate conversion to data manager
            count, warnings = self.data_manager.batch_convert_actions_recursive(item)
            if count > 0:
                msg = f"{tr('Converted')} {count} {tr('actions to commands.')}"
                if warnings > 0:
                    msg += f"\n\n{tr('Warnings')}: {warnings} {tr('items require attention.')}\n"
                    msg += tr("(Look for items marked with 'Legacy Warning')")
                    QMessageBox.warning(self, tr("Conversion Complete"), msg)
                else:
                    QMessageBox.information(self, tr("Conversion Complete"), msg)

    def replace_item_with_command(self, index, cmd_data):
        """UI wrapper to replace a legacy Action item with a new Command item via DataManager."""
        cmd_item = self.data_manager.replace_action_with_command(index, cmd_data)
        if cmd_item:
            # Update Selection and Expansion
            self.setCurrentIndex(cmd_item.index())
            self.expand(cmd_item.index()) # Expand to show preserved children

    def add_keywords(self, parent_index):
        if not parent_index.isValid(): return
        self.add_child_item(parent_index, "KEYWORDS", {}, tr("Keywords"))

    def add_trigger(self, parent_index):
        if not parent_index.isValid(): return
        eff_data = self.data_manager.create_default_trigger_data()
        self.add_child_item(parent_index, "EFFECT", eff_data, f"{tr('Effect')}: {tr('ON_PLAY')}")

    def add_static(self, parent_index):
        if not parent_index.isValid(): return
        mod_data = self.data_manager.create_default_static_data()
        self.add_child_item(parent_index, "MODIFIER", mod_data, f"{tr('Static')}: COST_MODIFIER")

    def add_reaction(self, parent_index):
        if not parent_index.isValid(): return
        ra_data = self.data_manager.create_default_reaction_data()
        self.add_child_item(parent_index, "REACTION_ABILITY", ra_data, f"{tr('Reaction Ability')}: NINJA_STRIKE")

    def add_option(self, parent_index):
        # NOTE: Deprecated for Actions, but useful for COMMAND 'CHOICE' structures if manually building.
        # But generally options are part of the command structure.
        if not parent_index.isValid(): return
        parent_item = self.standard_model.itemFromIndex(parent_index)
        if parent_item is None:
            return
        count = parent_item.rowCount() + 1

        new_item = QStandardItem(f"{tr('Option')} {count}")
        new_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
        new_item.setData({'uid': str(uuid.uuid4())}, Qt.ItemDataRole.UserRole + 2)

        parent_item.appendRow(new_item)
        self.expand(parent_index)
        self.setCurrentIndex(new_item.index())

    def add_command_to_option(self, option_index, cmd_data=None):
        if not option_index.isValid(): return
        data_copy = self.data_manager.create_default_command_data(cmd_data)
        label = self.data_manager.format_command_label(data_copy)
        self.add_child_item(option_index, "COMMAND", data_copy, label)

    def add_action_sibling(self, action_index, action_data=None):
        pass

    def add_command_to_effect(self, effect_index, cmd_data=None):
        if not effect_index.isValid(): return
        data_copy = self.data_manager.create_default_command_data(cmd_data)
        label = self.data_manager.format_command_label(data_copy)
        self.add_child_item(effect_index, "COMMAND", data_copy, label)

    def add_action_to_effect(self, effect_index, action_data=None):
        # Redirect to add_command_to_effect with default
        if action_data is None:
             self.add_command_to_effect(effect_index)
             return
        # If data provided, try to convert on fly
        try:
             from dm_toolkit.gui.editor.action_converter import ActionConverter
             cmd_data = ActionConverter.convert(action_data)
             self.add_command_to_effect(effect_index, cmd_data)
        except Exception:
             pass

    def add_action_to_option(self, option_index, action_data=None):
        if action_data is None:
             self.add_command_to_option(option_index)
             return
        try:
             from dm_toolkit.gui.editor.action_converter import ActionConverter
             cmd_data = ActionConverter.convert(action_data)
             self.add_command_to_option(option_index, cmd_data)
        except Exception:
             pass

    def add_command_contextual(self, cmd_data=None):
        idx = self.currentIndex()
        if not idx.isValid(): return

        item = self.standard_model.itemFromIndex(idx)
        if item is None:
            return
        type_ = self.data_manager.get_item_type(item)

        if type_ == "EFFECT":
            self.add_command_to_effect(idx, cmd_data)
        elif type_ == "OPTION":
            self.add_command_to_option(idx, cmd_data)
        elif type_ in ["COMMAND", "ACTION"]:
             # Sibling or Branch
             parent = item.parent()
             if parent:
                 parent_type = self.data_manager.get_item_type(parent)
                 if parent_type == "EFFECT":
                     self.add_command_to_effect(parent.index(), cmd_data)
                 elif parent_type == "OPTION":
                     self.add_command_to_option(parent.index(), cmd_data)
                 elif parent_type in ["CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
                     self._add_command_to_branch(parent.index(), cmd_data)
        elif type_ in ["CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
             self._add_command_to_branch(idx, cmd_data)

    def add_action_contextual(self, action_data=None):
        self.add_command_contextual(action_data)

    def _add_command_to_branch(self, branch_index, cmd_data=None):
        if not branch_index.isValid(): return
        data_copy = self.data_manager.create_default_command_data(cmd_data)
        self.add_child_item(branch_index, "COMMAND", data_copy, f"{tr('Action')}: {tr(data_copy.get('type', 'NONE'))}")

    def generate_branches_for_current(self):
        """Generates child branches for the currently selected command item."""
        index = self.currentIndex()
        if not index.isValid(): return

        item = self.standard_model.itemFromIndex(index)
        if item is None:
            return
        item_type = self.data_manager.get_item_type(item)

        if item_type == "COMMAND":
            self.data_manager.add_command_branches(item)
            self.expand(index)

    def add_effect_interactive(self, parent_index):
        if not parent_index.isValid(): return

        parent_item = self.standard_model.itemFromIndex(parent_index)
        if parent_item is None:
            return
        role = self.data_manager.get_item_type(parent_item)

        items = [tr("Triggered Ability"), tr("Static Ability")]

        # Check if we can add Reaction Ability
        # Only for CARD (not SPELL_SIDE)
        if role == "CARD":
             card_type = self.data_manager.get_card_context_type(parent_item)
             if card_type != "SPELL":
                  items.append(tr("Reaction Ability"))

        item, ok = QInputDialog.getItem(self, tr("Add Effect"), tr("Select Effect Type"), items, 0, False)

        if ok and item:
            if item == tr("Triggered Ability"):
                eff_data = self.data_manager.create_default_trigger_data()
                self.add_child_item(parent_index, "EFFECT", eff_data, f"{tr('Effect')}: ON_PLAY")
            elif item == tr("Static Ability"):
                mod_data = self.data_manager.create_default_static_data()
                self.add_child_item(parent_index, "MODIFIER", mod_data, f"{tr('Static')}: COST_MODIFIER")
            elif item == tr("Reaction Ability"):
                self.add_reaction(parent_index)

    def move_effect_item(self, item, target_type):
        """Updates the item's visual state (Label) and Type to match the new effect type."""
        self.data_manager.update_effect_type(item, target_type)

    def load_data(self, cards_data):
        # Save Expansion State
        expanded_ids = self._save_expansion_state()

        self.data_manager.load_data(cards_data)

        # Restore Expansion State
        self._restore_expansion_state(expanded_ids)

    def _save_expansion_state(self):
        """Saves the IDs of expanded items."""
        expanded_ids: set[str] = set()
        root = self.standard_model.invisibleRootItem()
        self._traverse_save_expansion(root, expanded_ids)
        return expanded_ids

    def _traverse_save_expansion(self, item, expanded_ids):
        index = item.index()
        # Root index is invalid but we traverse its children
        if index.isValid() and self.isExpanded(index):
            # Use data_manager helper
            path = self.data_manager.get_item_path(item)
            expanded_ids.add(path)
        for i in range(item.rowCount()):
            child = item.child(i)
            if child is None:
                continue
            self._traverse_save_expansion(child, expanded_ids)

    def _restore_expansion_state(self, expanded_ids):
        """Restores expansion state based on saved paths."""
        root = self.standard_model.invisibleRootItem()
        self._traverse_restore_expansion(root, expanded_ids)

    def _traverse_restore_expansion(self, item, expanded_ids):
        index = item.index()
        if index.isValid():
            # Use data_manager helper
            path = self.data_manager.get_item_path(item)
            if path in expanded_ids:
                self.setExpanded(index, True)
        for i in range(item.rowCount()):
            child = item.child(i)
            if child is None:
                continue
            self._traverse_restore_expansion(child, expanded_ids)

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
            # After structural change, immediately reconstruct the card data and update the card_item's stored dict
            try:
                updated = self.data_manager.reconstruct_card_data(card_item)
                if updated:
                    self.data_manager.set_item_data(card_item, updated)
            except Exception:
                pass
            self.setCurrentIndex(eff_item.index())
            self.expand(card_index)
        return eff_item

    def remove_rev_change(self, card_index):
        if not card_index.isValid(): return
        card_item = self.standard_model.itemFromIndex(card_index)
        self.data_manager.remove_revolution_change_logic(card_item)
        try:
            updated = self.data_manager.reconstruct_card_data(card_item)
            if updated:
                self.data_manager.set_item_data(card_item, updated)
        except Exception:
            pass
