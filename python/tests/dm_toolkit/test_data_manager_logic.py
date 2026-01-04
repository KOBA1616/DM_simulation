
import unittest
from unittest.mock import MagicMock, patch
import sys
import types

# Mock PyQt6 modules before importing DataManager
m_qt = types.ModuleType('PyQt6')
m_widgets = types.ModuleType('PyQt6.QtWidgets')
m_gui = types.ModuleType('PyQt6.QtGui')
m_core = types.ModuleType('PyQt6.QtCore')

sys.modules['PyQt6'] = m_qt
sys.modules['PyQt6.QtWidgets'] = m_widgets
sys.modules['PyQt6.QtGui'] = m_gui
sys.modules['PyQt6.QtCore'] = m_core

# Mock QStandardItem and Model
class MockItem:
    def __init__(self, text=""):
        self._text = text
        self._data = {}
        self._rows = []
        self._parent = None

    def text(self): return self._text
    def setText(self, t): self._text = t

    def data(self, role): return self._data.get(role)
    def setData(self, value, role): self._data[role] = value

    def rowCount(self): return len(self._rows)
    def child(self, row):
        if 0 <= row < len(self._rows): return self._rows[row]
        return None

    def appendRow(self, item):
        item._parent = self
        self._rows.append(item)

    def insertRow(self, row, item):
        item._parent = self
        self._rows.insert(row, item)

    def removeRow(self, row):
        if 0 <= row < len(self._rows):
            self._rows.pop(row)

    def parent(self): return self._parent

    def index(self):
        return MagicMock()

    def row(self):
        # Infer row from parent
        if self._parent:
            return self._parent._rows.index(self)
        return 0

    def setEditable(self, val): pass
    def setToolTip(self, val): pass

class MockModel:
    def __init__(self):
        self._root = MockItem("ROOT")

    def invisibleRootItem(self): return self._root

    def itemFromIndex(self, index):
        # Simplification: we might need to rely on mocking return values in tests
        return None

    def clear(self):
        self._root = MockItem("ROOT")

    def setHorizontalHeaderLabels(self, labels): pass

    def appendRow(self, item):
        self._root.appendRow(item)

m_gui.QStandardItemModel = MockModel
m_gui.QStandardItem = MockItem
m_core.Qt = MagicMock()
m_core.Qt.ItemDataRole.UserRole = 256

class MockQModelIndex:
    def isValid(self): return True
    def row(self): return 0
    def parent(self): return MockQModelIndex()

m_core.QModelIndex = MockQModelIndex

# Now we can import DataManager
# We also need to mock dm_toolkit.gui.localization
sys.modules['dm_toolkit.gui.localization'] = MagicMock()
sys.modules['dm_toolkit.gui.localization'].tr = lambda x: x

from dm_toolkit.gui.editor.data_manager import CardDataManager

class TestDataManagerMocked(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.manager = CardDataManager(self.model)
        # Mock templates to avoid file IO
        self.manager.templates = {"commands": [], "actions": []}

    def test_create_action_conversion(self):
        # Create a legacy action data
        action_data = {
            "type": "DRAW_CARD",
            "value": 1
        }

        # We need to mock ActionConverter because it's imported in DataManager
        # But DataManager imports it at top level.
        # Since we already imported DataManager, it has already imported ActionConverter.
        # We can patch it.

        with patch('dm_toolkit.gui.editor.data_manager.convert_action_to_objs') as mock_conv:
            # Setup mock return
            mock_cmd = MagicMock()
            mock_cmd.to_dict.return_value = {
                "type": "TRANSITION",
                "to_zone": "HAND",
                "amount": 1
            }
            # The function returns a list of objects
            mock_conv.return_value = [mock_cmd]

            # This should trigger conversion in add_child_item if legacy save is off
            # Ensure env var is not set
            import os
            if 'EDITOR_LEGACY_SAVE' in os.environ: del os.environ['EDITOR_LEGACY_SAVE']

            root = self.model.invisibleRootItem()
            # We need a valid index for add_child_item.
            # But add_child_item calls model.itemFromIndex(parent_index).
            # We need to mock that interaction.

            # Actually, add_child_item implementation:
            # parent_item = self.model.itemFromIndex(parent_index)

            # Let's bypass add_child_item and test _create_action_item directly
            # or test logic that uses conversion.

            # Let's test `convert_action_tree_to_command` which I added.

            # Setup an item structure that represents an ACTION
            action_item = MockItem("Action: DRAW_CARD")
            action_item.setData("ACTION", 257) # UserRole + 1
            action_item.setData(action_data, 258) # UserRole + 2

            # Mock ActionConverter.convert
            # Since convert_action_tree_to_command imports locally, we must patch where it imports from
            with patch('dm_toolkit.gui.editor.action_converter.ActionConverter') as MockConverter:
                MockConverter.convert.return_value = {
                    "type": "TRANSITION",
                    "to_zone": "HAND",
                    "amount": 1
                }

                result = self.manager.convert_action_tree_to_command(action_item)

                self.assertEqual(result['type'], "TRANSITION")
                self.assertEqual(result['amount'], 1)

    def test_collect_conversion_preview(self):
        # Create a tree: ROOT -> ACTION
        action_item = MockItem("Action: DRAW_CARD")
        action_item.setData("ACTION", 257)
        action_item.setData({"type": "DRAW_CARD"}, 258)

        root = self.model.invisibleRootItem()
        root.appendRow(action_item)

        with patch('dm_toolkit.gui.editor.action_converter.ActionConverter') as MockConverter:
             MockConverter.convert.return_value = {"type": "TRANSITION"}

             previews = self.manager.collect_conversion_preview(root)

             self.assertEqual(len(previews), 1)
             self.assertEqual(previews[0]['label'], "Action: DRAW_CARD")
             self.assertEqual(previews[0]['cmd_data']['type'], "TRANSITION")

if __name__ == '__main__':
    unittest.main()
