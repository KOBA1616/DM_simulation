# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QTreeView, QAbstractItemView, QInputDialog, QMessageBox
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QModelIndex
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.data_manager import CardDataManager
from dm_toolkit.gui.editor.context_menus import LogicTreeContextMenuHandler
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.qt_impl import QtEditorModel, QtEditorItem
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

        # Initialize Data Manager
        self.qt_model = QtEditorModel(self.standard_model)
        self.data_manager = CardDataManager(self.qt_model)

        # Initialize Context Menu Handler
        self.context_menu_handler = LogicTreeContextMenuHandler(self)
        self.customContextMenuRequested.connect(self.context_menu_handler.show_context_menu)

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
        # Resolve index via data manager
        new_item = self.data_manager.add_reaction(parent_index)
        if new_item:
            self.setExpanded(parent_index, True)
            if isinstance(new_item, QtEditorItem):
                self.setCurrentIndex(new_item.get_raw_item().index())

    def add_option(self, parent_index):
        # NOTE: Deprecated for Actions, but useful for COMMAND 'CHOICE' structures if manually building.
        # But generally options are part of the command structure.
        if not parent_index.isValid(): return
        parent_item = self.standard_model.itemFromIndex(parent_index)
        if parent_item is None:
            return
        count = parent_item.rowCount() + 1

        new_item = QStandardItem(f"{tr('Option')} {count}")
        new_item.setData("OPTION", ROLE_TYPE)
        new_item.setData({'uid': str(uuid.uuid4())}, ROLE_DATA)

        parent_item.appendRow(new_item)
        self.expand(parent_index)
        self.setCurrentIndex(new_item.index())

    def add_command_to_option(self, option_index, cmd_data=None):
        # Delegate to DataManager via contextual add
        new_item = self.data_manager.add_command_contextual(option_index, cmd_data)
        if new_item and isinstance(new_item, QtEditorItem):
            self.setExpanded(new_item.parent().get_raw_item().index(), True)
            self.setCurrentIndex(new_item.get_raw_item().index())

    def add_action_sibling(self, action_index, action_data=None):
        pass

    def add_command_to_effect(self, effect_index, cmd_data=None):
        # Delegate to DataManager via contextual add
        new_item = self.data_manager.add_command_contextual(effect_index, cmd_data)
        if new_item and isinstance(new_item, QtEditorItem):
            self.setExpanded(new_item.parent().get_raw_item().index(), True)
            self.setCurrentIndex(new_item.get_raw_item().index())

    def add_action_to_effect(self, effect_index, action_data=None):
        self.add_command_to_effect(effect_index, action_data)

    def add_action_to_option(self, option_index, action_data=None):
        self.add_command_to_option(option_index, action_data)

    def add_command_contextual(self, cmd_data=None):
        idx = self.currentIndex()
        if not idx.isValid(): return

        new_item = self.data_manager.add_command_contextual(idx, cmd_data)
        if new_item and isinstance(new_item, QtEditorItem):
            self.setExpanded(new_item.parent().get_raw_item().index(), True)
            self.setCurrentIndex(new_item.get_raw_item().index())

    def add_action_contextual(self, action_data=None):
        self.add_command_contextual(action_data)

    def _add_command_to_branch(self, branch_index, cmd_data=None):
        # Delegate to DataManager via contextual add
        new_item = self.data_manager.add_command_contextual(branch_index, cmd_data)
        if new_item and isinstance(new_item, QtEditorItem):
             self.setExpanded(new_item.parent().get_raw_item().index(), True)
             self.setCurrentIndex(new_item.get_raw_item().index())

    def generate_branches_for_current(self):
        """Generates child branches for the currently selected command item."""
        index = self.currentIndex()
        if not index.isValid(): return

        # Pass index directly
        item_type = self.data_manager.get_item_type(index)

        if item_type == "COMMAND":
            self.data_manager.add_command_branches(index)
            self.expand(index)

    def add_effect_interactive(self, parent_index):
        if not parent_index.isValid(): return

        # Check role via data manager using index
        role = self.data_manager.get_item_type(parent_index)

        items = [tr("Triggered Ability"), tr("Static Ability")]

        # Check if we can add Reaction Ability
        # Only for CARD (not SPELL_SIDE)
        if role == "CARD":
             card_type = self.data_manager.get_card_context_type(parent_index)
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
        # item is QStandardItem
        # Update using wrapper
        wrapped = QtEditorItem(item)
        self.data_manager.update_effect_type(wrapped, target_type)

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
            # Use data_manager helper, pass wrapper
            path = self.data_manager.get_item_path(QtEditorItem(item))
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
            path = self.data_manager.get_item_path(QtEditorItem(item))
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
        if item and isinstance(item, QtEditorItem):
            self.setCurrentIndex(item.get_raw_item().index())
            self.expand(item.get_raw_item().index())
        return item

    def add_child_item(self, parent_index, item_type, data, label):
        new_item = self.data_manager.add_child_item(parent_index, item_type, data, label)
        if new_item and isinstance(new_item, QtEditorItem):
            self.setExpanded(parent_index, True)
            self.setCurrentIndex(new_item.get_raw_item().index())
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
        item = self.data_manager.add_spell_side_item(card_index)
        if item and isinstance(item, QtEditorItem):
            self.setCurrentIndex(item.get_raw_item().index())
            self.expand(card_index)
        return item

    def remove_spell_side(self, card_index):
        if not card_index.isValid(): return
        self.data_manager.remove_spell_side_item(card_index)

    def add_rev_change(self, card_index):
        if not card_index.isValid(): return
        eff_item = self.data_manager.apply_template_by_key(card_index, "REVOLUTION_CHANGE", "Revolution Change")
        if eff_item and isinstance(eff_item, QtEditorItem):
            self.setCurrentIndex(eff_item.get_raw_item().index())
            self.expand(card_index)
        return eff_item

    def remove_rev_change(self, card_index):
        if not card_index.isValid(): return
        self.data_manager.remove_logic_by_label(card_index, "Revolution Change")

    def add_mekraid(self, card_index):
        """メクレイド効果を追加"""
        if not card_index.isValid(): return
        eff_item = self.data_manager.apply_template_by_key(card_index, "MEKRAID", "Mekraid")
        if eff_item and isinstance(eff_item, QtEditorItem):
            self.setCurrentIndex(eff_item.get_raw_item().index())
            self.expand(card_index)
        return eff_item

    def remove_mekraid(self, card_index):
        """メクレイド効果を削除"""
        if not card_index.isValid(): return
        self.data_manager.remove_logic_by_label(card_index, "Mekraid")

    def add_friend_burst(self, card_index):
        """フレンド・バースト効果を追加"""
        if not card_index.isValid(): return
        eff_item = self.data_manager.apply_template_by_key(card_index, "FRIEND_BURST", "Friend Burst")
        if eff_item and isinstance(eff_item, QtEditorItem):
            self.setCurrentIndex(eff_item.get_raw_item().index())
            self.expand(card_index)
        return eff_item

    def remove_friend_burst(self, card_index):
        """フレンド・バースト効果を削除"""
        if not card_index.isValid(): return
        self.data_manager.remove_logic_by_label(card_index, "Friend Burst")

    def add_mega_last_burst(self, card_index):
        """メガ・ラスト・バースト効果を追加"""
        if not card_index.isValid(): return
        eff_item = self.data_manager.apply_template_by_key(card_index, "MEGA_LAST_BURST", "Mega Last Burst")
        if eff_item and isinstance(eff_item, QtEditorItem):
            self.setCurrentIndex(eff_item.get_raw_item().index())
            self.expand(card_index)
        return eff_item

    def remove_mega_last_burst(self, card_index):
        """メガ・ラスト・バースト効果を削除"""
        if not card_index.isValid(): return
        self.data_manager.remove_logic_by_label(card_index, "Mega Last Burst")

    def request_generate_options(self):
        if not getattr(self, 'current_item', None):
            return
        try:
            count = int(self.option_count_spin.value())
        except Exception:
            count = 1

        wrapped = QtEditorItem(self.current_item)
        self.data_manager.generate_options(wrapped, count)
        self.expand(wrapped.get_raw_item().index())
