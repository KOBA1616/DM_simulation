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

        # Initialize Data Manager
        self.data_manager = CardDataManager(self.model)

        self.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if not indexes:
            return
        index = indexes[0]
        # Just ensure generic handling if needed
        pass

    def mousePressEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid():
             item_type = index.data(Qt.ItemDataRole.UserRole + 1)
             if item_type == "CARD":
                 if self.isExpanded(index):
                     self.collapse(index)
                 else:
                     self.expandRecursively(index)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        index = self.indexAt(event.pos())
        if not index.isValid():
            menu = QMenu(self)
            add_card = QAction(tr("Add New Card"), self)
            add_card.triggered.connect(self.add_new_card)
            menu.addAction(add_card)
            menu.exec(event.globalPos())
            return

        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        menu = QMenu(self)

        if item_type == "CARD":
            add_effect = QAction(tr("Add Effect"), self)
            add_effect.triggered.connect(lambda: self.add_child_item(index, "EFFECT", {"trigger": "NONE", "actions": []}, tr("New Effect")))
            menu.addAction(add_effect)

            add_reaction = QAction(tr("Add Reaction"), self)
            add_reaction.triggered.connect(lambda: self.add_child_item(index, "REACTION", {"type": "NONE", "zone": "HAND"}, tr("New Reaction")))
            menu.addAction(add_reaction)

        elif item_type == "EFFECT":
            add_action = QAction(tr("Add Action"), self)
            add_action.triggered.connect(lambda: self.add_child_item(index, "ACTION", {"type": "NONE"}, tr("New Action")))
            menu.addAction(add_action)

        menu.addSeparator()
        remove_action = QAction(tr("Remove Item"), self)
        remove_action.triggered.connect(self.remove_current_item)
        menu.addAction(remove_action)

        menu.exec(event.globalPos())

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
