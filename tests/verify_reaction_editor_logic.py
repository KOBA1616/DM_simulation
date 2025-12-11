import sys
import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt

# Adjust path
sys.path.append('.')

from dm_toolkit.gui.editor.data_manager import CardDataManager

class TestReactionLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create application for Qt signals
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def test_load_and_save_reaction(self):
        model = QStandardItemModel()
        manager = CardDataManager(model)

        # Mock Data
        original_data = [{
            "id": 1000,
            "name": "Test Ninja",
            "type": "CREATURE",
            "reaction_abilities": [
                {
                    "type": "NINJA_STRIKE",
                    "cost": 7,
                    "zone": "HAND",
                    "condition": {
                        "trigger_event": "ON_BLOCK_OR_ATTACK",
                        "mana_count_min": 7,
                        "civilization_match": True
                    }
                }
            ],
            "effects": []
        }]

        # 1. Load Data
        manager.load_data(original_data)

        # 2. Verify Model Structure
        root = model.invisibleRootItem()
        self.assertEqual(root.rowCount(), 1)
        card_item = root.child(0)
        self.assertEqual(card_item.rowCount(), 1) # 1 reaction

        reaction_item = card_item.child(0)
        item_type = reaction_item.data(Qt.ItemDataRole.UserRole + 1)
        self.assertEqual(item_type, "REACTION")

        reaction_data = reaction_item.data(Qt.ItemDataRole.UserRole + 2)
        self.assertEqual(reaction_data['type'], "NINJA_STRIKE")
        self.assertEqual(reaction_data['cost'], 7)

        # 3. Save (Reconstruct) Data
        saved_data = manager.get_full_data()
        self.assertEqual(len(saved_data), 1)
        saved_card = saved_data[0]
        self.assertIn('reaction_abilities', saved_card)
        self.assertEqual(len(saved_card['reaction_abilities']), 1)

        saved_reaction = saved_card['reaction_abilities'][0]
        self.assertEqual(saved_reaction['type'], "NINJA_STRIKE")
        self.assertEqual(saved_reaction['condition']['trigger_event'], "ON_BLOCK_OR_ATTACK")

    def test_add_reaction_programmatically(self):
        model = QStandardItemModel()
        manager = CardDataManager(model)

        # Add Card
        card_item = manager.add_new_card()

        # Add Reaction via add_child_item (LogicTreeWidget uses this)
        react_data = {"type": "STRIKE_BACK", "zone": "HAND", "cost": 0}
        manager.add_child_item(card_item.index(), "REACTION", react_data, "New Reaction")

        # Verify
        saved_data = manager.get_full_data()
        self.assertEqual(len(saved_data[0]['reaction_abilities']), 1)
        self.assertEqual(saved_data[0]['reaction_abilities'][0]['type'], "STRIKE_BACK")

if __name__ == '__main__':
    unittest.main()
