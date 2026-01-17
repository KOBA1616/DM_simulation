
import unittest
from unittest.mock import MagicMock, patch

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

from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.data_manager import CardDataManager

class TestDataManagerMocked(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.manager = CardDataManager(self.model)

    def test_create_action_conversion(self):
        # Create a legacy action data
        action_data = {
            "type": "DRAW_CARD",
            "value": 1
        }

        # Setup an item structure that represents an ACTION
        action_item = MockItem("Action: DRAW_CARD")
        action_item.setData("ACTION", ROLE_TYPE)  # UserRole + 1
        action_item.setData(action_data, ROLE_DATA)  # UserRole + 2

        # Mock ActionConverter.convert
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
        action_item.setData("ACTION", ROLE_TYPE)
        action_item.setData({"type": "DRAW_CARD"}, ROLE_DATA)

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
