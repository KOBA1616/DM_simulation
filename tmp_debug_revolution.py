from unittest.mock import MagicMock, patch
from typing import Any, cast
import sys
# Mock PyQt6
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
from dm_toolkit.gui.editor.data_manager import CardDataManager
from dm_toolkit.gui.editor.data_manager import QStandardItem, Qt
import uuid

# Setup mock model
mock_model = MagicMock()
root_item = QStandardItem("Root")
mock_model.invisibleRootItem.return_value = root_item

manager = CardDataManager(mock_model)

# Create card item like test
card_data = {"id": 1, "name": "Rev Change Unit", "keywords": {}}
card_item = QStandardItem("Card")
card_item.setData("CARD", 256 + 1)
card_item.setData(card_data, 256 + 2)
trig_group = QStandardItem("Trigger Group")
trig_group.setData("GROUP_TRIGGER", 256 + 1)
card_item.appendRow(trig_group)
root_item.appendRow(card_item)

manager.add_revolution_change_logic(card_item)

print('Tree structure:')
print('card_item.rowCount=', card_item.rowCount())
eff_item = trig_group.child(0)
print('eff_item.role=', eff_item.data(256+1))
print('eff_item.data=', eff_item.data(256+2))
print('eff children count=', eff_item.rowCount())
child0 = eff_item.child(0)
print('child0.role=', child0.data(256+1))
print('child0.data=', child0.data(256+2))

out = manager.get_full_data()
print('get_full_data ->', out)
