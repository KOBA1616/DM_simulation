
import sys
import os
import json
import unittest
import pytest
from unittest.mock import MagicMock
from typing import Any, List, Dict, Optional

# --- Mocking PyQt6 and GUI dependencies ---
# This must happen BEFORE importing dm_toolkit.gui.editor.data_manager

class MockQStandardItem:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.rows: List['MockQStandardItem'] = []
        self._data: Dict[int, Any] = {}
        self.parent_item: Optional['MockQStandardItem'] = None

    def appendRow(self, item: 'MockQStandardItem') -> None:
        item.parent_item = self
        self.rows.append(item)

    def rowCount(self) -> int:
        return len(self.rows)

    def child(self, row: int) -> Optional['MockQStandardItem']:
        if 0 <= row < len(self.rows):
            return self.rows[row]
        return None

    def removeRow(self, row: int) -> None:
        if 0 <= row < len(self.rows):
            del self.rows[row]

    def data(self, role: int) -> Any:
        return self._data.get(role)

    def setData(self, value: Any, role: int) -> None:
        self._data[role] = value

    def setEditable(self, val: bool) -> None:
        pass

class MockQStandardItemModel:
    def __init__(self) -> None:
        self.root = MockQStandardItem("ROOT")

    def clear(self) -> None:
        self.root = MockQStandardItem("ROOT")

    def setHorizontalHeaderLabels(self, labels: List[str]) -> None:
        pass

    def appendRow(self, item: MockQStandardItem) -> None:
        self.root.appendRow(item)

    def invisibleRootItem(self) -> MockQStandardItem:
        return self.root

    def itemFromIndex(self, index: Any) -> Optional[MockQStandardItem]:
        return None

class MockQt:
    class ItemDataRole:
        UserRole = 256
        ForegroundRole = 0

# Mock sys.modules
if "PyQt6" not in sys.modules:
    sys.modules["PyQt6"] = MagicMock()
    sys.modules["PyQt6.QtGui"] = MagicMock()
    sys.modules["PyQt6.QtCore"] = MagicMock()

    sys.modules["PyQt6.QtGui"].QStandardItemModel = MockQStandardItemModel
    sys.modules["PyQt6.QtGui"].QStandardItem = MockQStandardItem
    sys.modules["PyQt6.QtCore"].Qt = MockQt

if "dm_toolkit.gui.localization" not in sys.modules:
    mock_loc = MagicMock()
    mock_loc.tr = lambda x: x
    sys.modules["dm_toolkit.gui.localization"] = mock_loc

# --- End Mocking ---

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

try:
    from dm_toolkit.gui.editor.data_manager import CardDataManager
except ImportError as e:
    # raise RuntimeError(f"Failed to import CardDataManager: {e}")
    CardDataManager = MagicMock() # Fallback for mypy check if file not found in path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
try:
    import dm_ai_module
    from dm_ai_module import Civilization, CardType
except ImportError:
    dm_ai_module = MagicMock()
    Civilization = MagicMock()
    CardType = MagicMock()

@pytest.mark.skipif("dm_ai_module" not in sys.modules, reason="dm_ai_module not found")
class TestGuiJsonIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MockQStandardItemModel()
        self.manager = CardDataManager(self.model)
        self.temp_file = "temp_gui_integration_test.json"

    def tearDown(self) -> None:
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_save_and_load_flow(self) -> None:
        """
        Verifies that data constructed via CardDataManager (simulating GUI)
        generates JSON that is valid and loadable by the C++ engine.
        """
        # 1. Simulate creating a new card in the GUI

        # Create Card Item
        card_data = {
            "id": 12345,
            "name": "GUI Test Dragon",
            "civilizations": ["FIRE"],
            "type": "CREATURE",
            "cost": 5,
            "power": 5000,
            "races": ["Armored Dragon"],
            "keywords": {"speed_attacker": True}
        }
        card_item = MockQStandardItem(f"{card_data['id']} - {card_data['name']}")
        card_item.setData("CARD", MockQt.ItemDataRole.UserRole + 1)
        card_item.setData(card_data, MockQt.ItemDataRole.UserRole + 2)

        # Add Keywords Item
        kw_item = MockQStandardItem("Keywords")
        kw_item.setData("KEYWORDS", MockQt.ItemDataRole.UserRole + 1)
        kw_item.setData({"speed_attacker": True}, MockQt.ItemDataRole.UserRole + 2)
        card_item.appendRow(kw_item)

        # Add Effect Item (Trigger: ON_PLAY)
        effect_data = {
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "actions": []
        }
        eff_item = MockQStandardItem("Effect: ON_PLAY")
        eff_item.setData("EFFECT", MockQt.ItemDataRole.UserRole + 1)
        eff_item.setData(effect_data, MockQt.ItemDataRole.UserRole + 2)

        # Add Action Item (DRAW_CARD)
        action_data = {
            "type": "DRAW_CARD",
            "filter": {
                "target_player": "PLAYER_SELF",
                "count": 1
            },
            "uid": "test-uid-1"
        }
        act_item = MockQStandardItem("Action: DRAW_CARD")
        act_item.setData("ACTION", MockQt.ItemDataRole.UserRole + 1)
        act_item.setData(action_data, MockQt.ItemDataRole.UserRole + 2)

        eff_item.appendRow(act_item)
        card_item.appendRow(eff_item)

        # Add to Model
        self.model.appendRow(card_item)

        # 2. Use DataManager to reconstruct (save) the data
        saved_data_list = self.manager.get_full_data()

        assert len(saved_data_list) == 1
        saved_card = saved_data_list[0]

        # Verify basic structure matches expectations
        assert saved_card["id"] == 12345
        assert saved_card["name"] == "GUI Test Dragon"
        assert saved_card["keywords"]["speed_attacker"] is True
        assert len(saved_card["effects"]) == 1
        assert saved_card["effects"][0]["actions"][0]["type"] == "DRAW_CARD"

        # 3. Write to JSON file
        with open(self.temp_file, "w") as f:
            json.dump(saved_data_list, f)

        # 4. Load with C++ Engine
        loaded_cards = dm_ai_module.JsonLoader.load_cards(self.temp_file)

        assert 12345 in loaded_cards
        engine_card = loaded_cards[12345]

        # 5. Verify Engine Data
        assert engine_card.name == "GUI Test Dragon"
        assert engine_card.cost == 5
        assert engine_card.power == 5000
        assert engine_card.civilizations[0] == Civilization.FIRE
        assert engine_card.type == CardType.CREATURE

        # Keyword access via nested struct
        assert engine_card.keywords.speed_attacker is True

        # Verify effect loaded (at least count)
        assert len(engine_card.effects) == 1

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
