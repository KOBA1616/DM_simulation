import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, cast

# Mock PyQt6 before importing dm_toolkit modules
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()

# Setup QStandardItem mock
class MockQStandardItem(MagicMock):
    def __init__(self, text="", **kwargs):
        super().__init__(**kwargs)
        self.text_val = text
        self._data = {}
        self._children = []
        self._parent = None

    def data(self, role):
        return self._data.get(role)

    def setData(self, value, role):
        self._data[role] = value

    def appendRow(self, item):
        self._children.append(item)
        item._parent = self

    def rowCount(self):
        return len(self._children)

    def child(self, row):
        if 0 <= row < len(self._children):
            return self._children[row]
        return None

    def removeRow(self, row):
        del self._children[row]

    def parent(self):
        return self._parent

    # Mocking setEditable to do nothing
    def setEditable(self, editable):
        pass

cast(Any, sys.modules['PyQt6.QtGui']).QStandardItem = MockQStandardItem
cast(Any, sys.modules['PyQt6.QtGui']).QStandardItemModel = MagicMock()
cast(Any, sys.modules['PyQt6.QtCore']).Qt = MagicMock()
cast(Any, sys.modules['PyQt6.QtCore']).Qt.ItemDataRole = MagicMock()
cast(Any, sys.modules['PyQt6.QtCore']).Qt.ItemDataRole.UserRole = 256
cast(Any, sys.modules['PyQt6.QtCore']).Qt.ItemDataRole.DisplayRole = 0

# Now import DataManager
from dm_toolkit.gui.editor.data_manager import CardDataManager

class TestCardEditorV2(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        # Mock invisibleRootItem
        self.root_item = MockQStandardItem("Root")
        self.mock_model.invisibleRootItem.return_value = self.root_item

        # When model.itemFromIndex is called, we need to return something valid if possible.
        # But DataManager mostly works with items directly or iterates rows.

        # Helper to bypass QStandardItemModel.itemFromIndex since we deal with items directly
        self.manager = CardDataManager(self.mock_model)

    def test_spell_side_roundtrip(self):
        """Test that Spell Side data is correctly loaded into Tree and reconstructed."""
        # Input Data
        input_data = [{
            "id": 1, "name": "Twinpact Creature", "type": "CREATURE",
            "spell_side": {
                "name": "Spell Side", "type": "SPELL", "cost": 3,
                "effects": [{"trigger": "ON_PLAY", "commands": []}]
            }
        }]

        # 1. Load Data
        def append_to_root(item):
            self.root_item.appendRow(item)
        self.mock_model.appendRow = append_to_root

        self.manager.load_data(input_data)

        # Verify Tree Structure
        self.assertEqual(self.root_item.rowCount(), 1)
        card_item = self.root_item.child(0)
        self.assertEqual(card_item.data(256 + 1), "CARD") # UserRole + 1

        # Check for Spell Side Child
        has_spell_side = False
        spell_side_item = None
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            # Check DisplayRole or UserRole for type
            if child.data(256 + 1) == "SPELL_SIDE":
                has_spell_side = True
                spell_side_item = child
                break

        self.assertTrue(has_spell_side, "Spell Side node should be created")

        # 2. Get Full Data (Reconstruct)
        # We need to make sure invisibleRootItem has the children we added
        # self.root_item already has them because we appended to it via append_to_root hook.

        output_data = self.manager.get_full_data()
        self.assertEqual(len(output_data), 1)
        out_card = output_data[0]

        self.assertIn("spell_side", out_card)
        self.assertEqual(out_card['spell_side']['name'], "Spell Side")
        self.assertEqual(len(out_card['spell_side']['effects']), 1)

    def test_add_revolution_change(self):
        """Test adding Revolution Change logic."""
        # Create a card item
        card_data = {"id": 1, "name": "Rev Change Unit", "keywords": {}}
        card_item = MockQStandardItem("Card")
        card_item.setData("CARD", 256 + 1)
        card_item.setData(card_data, 256 + 2)

        # IMPORTANT: The CardDataManager expects the CARD item to have 3 children:
        # 1. Keywords
        # 2. Group Trigger
        # 3. Group Static
        # 4. Group Reaction
        # We must manually create them to simulate a valid card item, OR call _create_card_item logic.

        # Let's manually add the structure expected by add_revolution_change_logic
        trig_group = MockQStandardItem("Trigger Group")
        trig_group.setData("GROUP_TRIGGER", 256 + 1)
        card_item.appendRow(trig_group)

        self.root_item.appendRow(card_item)

        # Call logic
        self.manager.add_revolution_change_logic(card_item)

        # Verify Structure
        # The card item now has 1 child (Trigger Group)
        self.assertEqual(card_item.rowCount(), 1)

        # The trigger group should have 1 child (the effect)
        self.assertEqual(trig_group.rowCount(), 1)

        eff_item = trig_group.child(0)
        self.assertEqual(eff_item.data(256 + 1), "EFFECT")
        eff_data = eff_item.data(256 + 2)
        self.assertEqual(eff_data['trigger'], "ON_ATTACK_FROM_HAND")

        self.assertEqual(eff_item.rowCount(), 1)
        act_item = eff_item.child(0)
        act_data = act_item.data(256 + 2)
        # add_revolution_change_logic uses 'MUTATE' command with 'mutation_kind': 'REVOLUTION_CHANGE'
        self.assertEqual(act_data['type'], "MUTATE")
        self.assertEqual(act_data['mutation_kind'], "REVOLUTION_CHANGE")

        # Verify Get Full Data sets keyword
        output_data = self.manager.get_full_data()
        out_card = output_data[0]
        self.assertTrue(out_card['keywords'].get('revolution_change'), "Should auto-set keyword")

if __name__ == '__main__':
    unittest.main()
