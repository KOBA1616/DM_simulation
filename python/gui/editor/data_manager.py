from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
from gui.localization import tr

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Handles loading, saving (reconstruction), and item creation (ID generation).
    """

    def __init__(self, model: QStandardItemModel):
        self.model = model

    def load_data(self, cards_data):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Logic Tree"])

        for card_idx, card in enumerate(cards_data):
            card_item = self._create_card_item(card)

            for eff_idx, effect in enumerate(card.get('effects', [])):
                eff_item = self._create_effect_item(effect)

                for act_idx, action in enumerate(effect.get('actions', [])):
                    act_item = self._create_action_item(action)
                    eff_item.appendRow(act_item)

                card_item.appendRow(eff_item)

            self.model.appendRow(card_item)

    def get_full_data(self):
        """Reconstructs the full JSON list from the tree model."""
        cards = []
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            card_item = root.child(i)
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
        new_id = self._generate_new_id()
        new_card = {
            "id": new_id, "name": "New Card", "civilization": "FIRE", "type": "CREATURE",
            "cost": 1, "power": 1000, "races": [], "effects": []
        }
        item = self._create_card_item(new_card)
        self.model.appendRow(item)
        return item

    def add_child_item(self, parent_index, item_type, data, label):
        if not parent_index.isValid(): return None
        parent_item = self.model.itemFromIndex(parent_index)

        new_item = QStandardItem(label)
        new_item.setData(item_type, Qt.ItemDataRole.UserRole + 1)
        new_item.setData(data, Qt.ItemDataRole.UserRole + 2)

        parent_item.appendRow(new_item)
        return new_item

    def _create_card_item(self, card):
        item = QStandardItem(f"{card.get('id')} - {card.get('name', 'No Name')}")
        item.setData("CARD", Qt.ItemDataRole.UserRole + 1)
        item.setData(card, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_effect_item(self, effect):
        trig = effect.get('trigger', 'NONE')
        item = QStandardItem(f"{tr('Effect')}: {tr(trig)}")
        item.setData("EFFECT", Qt.ItemDataRole.UserRole + 1)
        item.setData(effect, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_action_item(self, action):
        # We could reproduce the sophisticated display text logic here or keep it simple
        # For consistency, let's replicate basic display logic or rely on update_data to fix it later?
        # Ideally, initial load should show correct text.
        act_type = action.get('type', 'NONE')
        # Simplified for initial load, Form updates will fix details if needed
        display_type = tr(act_type)
        if act_type == "GET_GAME_STAT":
             display_type = f"{tr('GET_GAME_STAT')} ({tr(action.get('str_val',''))})"
        elif act_type == "APPLY_MODIFIER" and action.get('str_val') == "COST":
             display_type = tr("COST_REDUCTION")
        elif act_type == "COST_REFERENCE":
             display_type = f"{tr('COST_REFERENCE')} ({tr(action.get('str_val',''))})"

        item = QStandardItem(f"{tr('Action')}: {display_type}")
        item.setData("ACTION", Qt.ItemDataRole.UserRole + 1)
        item.setData(action, Qt.ItemDataRole.UserRole + 2)
        return item

    def _generate_new_id(self):
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
        return max_id + 1
