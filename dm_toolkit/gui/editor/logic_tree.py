from PyQt6.QtWidgets import QTreeView, QAbstractItemView, QMenu
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
        pass

    def mousePressEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid() and event.button() == Qt.MouseButton.LeftButton:
             item_type = index.data(Qt.ItemDataRole.UserRole + 1)
             if item_type in ["CARD", "SPELL_SIDE"]:
                 if self.isExpanded(index):
                     self.collapse(index)
                 else:
                     self.expandRecursively(index)
        super().mousePressEvent(event)

    def show_context_menu(self, pos):
        index = self.indexAt(pos)
        if not index.isValid():
            return

        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        menu = QMenu(self)

        if item_type == "ACTION":
            act_data = index.data(Qt.ItemDataRole.UserRole + 2)
            if act_data.get('type') == "SELECT_OPTION":
                add_opt_action = QAction(tr("Add Option"), self)
                add_opt_action.triggered.connect(lambda: self.add_option(index))
                menu.addAction(add_opt_action)

        elif item_type == "OPTION":
            add_act_action = QAction(tr("Add Action"), self)
            add_act_action.triggered.connect(lambda: self.add_action_to_option(index))
            menu.addAction(add_act_action)

            remove_opt_action = QAction(tr("Remove Option"), self)
            remove_opt_action.triggered.connect(lambda: self.remove_current_item())
            menu.addAction(remove_opt_action)

        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(pos))

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

        # Default Action
        act_data = {"type": "NONE"}

        self.add_child_item(option_index, "ACTION", act_data, f"{tr('Action')}: NONE")

    def load_data(self, cards_data):
        self.data_manager.load_data(cards_data)

    def get_full_data_from_model(self):
        return self.data_manager.get_full_data()

    def add_new_card(self):
        item = self.data_manager.add_new_card()
        if item:
            self.setCurrentIndex(item.index())
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
