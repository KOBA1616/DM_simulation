from PyQt6.QtWidgets import QTreeView, QAbstractItemView
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QModelIndex
from gui.localization import tr

class LogicTreeWidget(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = QStandardItemModel()
        self.setModel(self.model)
        self.setHeaderHidden(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

        # Signals
        # We rely on selectionModel().selectionChanged for external updates
        self.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if not indexes:
            return

        # We only care about the first selected item in SingleSelection mode
        # Note: selected.indexes() returns all selected cells (columns). We just check the first one.
        index = indexes[0]

        # Check if it's a CARD
        item_type = index.data(Qt.ItemDataRole.UserRole + 1)
        if item_type == "CARD":
            self.expandRecursively(index)

    def load_data(self, cards_data):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Logic Tree"])

        for card_idx, card in enumerate(cards_data):
            card_item = QStandardItem(f"{card.get('id')} - {card.get('name', 'No Name')}")
            card_item.setData("CARD", Qt.ItemDataRole.UserRole + 1) # Type
            card_item.setData(card, Qt.ItemDataRole.UserRole + 2) # Data Reference (copy)
            card_item.setData(card_idx, Qt.ItemDataRole.UserRole + 3) # Original Index

            for eff_idx, effect in enumerate(card.get('effects', [])):
                trig = effect.get('trigger', 'NONE')
                eff_item = QStandardItem(f"{tr('Effect')}: {tr(trig)}")
                eff_item.setData("EFFECT", Qt.ItemDataRole.UserRole + 1)
                eff_item.setData(effect, Qt.ItemDataRole.UserRole + 2)

                for act_idx, action in enumerate(effect.get('actions', [])):
                    act_type = action.get('type', 'NONE')
                    act_item = QStandardItem(f"{tr('Action')}: {tr(act_type)}")
                    act_item.setData("ACTION", Qt.ItemDataRole.UserRole + 1)
                    act_item.setData(action, Qt.ItemDataRole.UserRole + 2)
                    eff_item.appendRow(act_item)

                card_item.appendRow(eff_item)

            self.model.appendRow(card_item)

    def get_full_data_from_model(self):
        """Reconstructs the full JSON list from the tree model."""
        cards = []
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            card_item = root.child(i)
            # We don't just take the stored data, we reconstruct it to capture hierarchy changes
            # But the stored data is the source of truth for fields not represented in the tree structure?
            # Actually, the Forms update the item's stored data.
            # So we iterate and collect.

            card_data = card_item.data(Qt.ItemDataRole.UserRole + 2)
            if not card_data: continue

            # Rebuild effects list from children
            new_effects = []
            for j in range(card_item.rowCount()):
                eff_item = card_item.child(j)
                eff_data = eff_item.data(Qt.ItemDataRole.UserRole + 2)

                # Rebuild actions list from children
                new_actions = []
                for k in range(eff_item.rowCount()):
                    act_item = eff_item.child(k)
                    act_data = act_item.data(Qt.ItemDataRole.UserRole + 2)
                    new_actions.append(act_data)

                eff_data['actions'] = new_actions
                new_effects.append(eff_data)

            card_data['effects'] = new_effects
            cards.append(card_data)

        return cards

    def add_new_card(self):
        # Auto-increment ID logic
        max_id = 0
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            card_item = root.child(i)
            card_data = card_item.data(Qt.ItemDataRole.UserRole + 2)
            if card_data and 'id' in card_data:
                try:
                    cid = int(card_data['id'])
                    if cid > max_id:
                        max_id = cid
                except ValueError:
                    pass

        new_id = max_id + 1

        new_card = {
            "id": new_id, "name": "New Card", "civilization": "FIRE", "type": "CREATURE",
            "cost": 1, "power": 1000, "races": [], "effects": []
        }

        item = QStandardItem(f"{new_card['id']} - {new_card['name']}")
        item.setData("CARD", Qt.ItemDataRole.UserRole + 1)
        item.setData(new_card, Qt.ItemDataRole.UserRole + 2)
        self.model.appendRow(item)
        self.setCurrentIndex(item.index())
        return item

    def add_child_item(self, parent_index, item_type, data, label):
        if not parent_index.isValid(): return
        parent_item = self.model.itemFromIndex(parent_index)
        new_item = QStandardItem(label)
        new_item.setData(item_type, Qt.ItemDataRole.UserRole + 1)
        new_item.setData(data, Qt.ItemDataRole.UserRole + 2)
        parent_item.appendRow(new_item)
        self.setExpanded(parent_index, True)
        self.setCurrentIndex(new_item.index())

    def remove_current_item(self):
        idx = self.currentIndex()
        if not idx.isValid(): return
        self.model.removeRow(idx.row(), idx.parent())
