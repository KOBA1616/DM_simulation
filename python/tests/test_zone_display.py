import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock PyQt6
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtWidgets"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()
sys.modules["PyQt6.QtGui"] = MagicMock()

# Mock Qt classes used in zone_widget.py
class MockQWidget:
    def __init__(self, parent=None):
        pass
    def setParent(self, parent):
        pass
    def window(self):
        return self

class MockSignal:
    def __init__(self):
        self.callbacks = []
    def connect(self, callback):
        self.callbacks.append(callback)
    def emit(self, *args):
        for cb in self.callbacks:
            cb(*args)

sys.modules["PyQt6.QtWidgets"].QWidget = MockQWidget
sys.modules["PyQt6.QtWidgets"].QHBoxLayout = MagicMock()
sys.modules["PyQt6.QtWidgets"].QLabel = MagicMock()
sys.modules["PyQt6.QtWidgets"].QScrollArea = MagicMock()
sys.modules["PyQt6.QtCore"].pyqtSignal = lambda *args: MockSignal()
sys.modules["PyQt6.QtCore"].Qt = MagicMock()

# Mock tr function
if "dm_toolkit.gui.i18n" not in sys.modules:
    sys.modules["dm_toolkit.gui.i18n"] = MagicMock()
    sys.modules["dm_toolkit.gui.i18n"].tr = lambda x: x

# Mock card_helpers
if "dm_toolkit.gui.utils.card_helpers" not in sys.modules:
    sys.modules["dm_toolkit.gui.utils.card_helpers"] = MagicMock()
    sys.modules["dm_toolkit.gui.utils.card_helpers"].get_card_civilization = lambda x: "FIRE"

# Mock commands
if "dm_toolkit.commands" not in sys.modules:
    sys.modules["dm_toolkit.commands"] = MagicMock()
    sys.modules["dm_toolkit.commands"].wrap_action = lambda x: x

# Mock CardWidget
mock_card_widget_module = MagicMock()
class MockCardWidget:
    def __init__(self, *args, **kwargs):
        self.clicked = MockSignal()
        self.hovered = MockSignal()
        self.action_triggered = MockSignal()
        self.double_clicked = MockSignal()
        self.instance_id = -1
        # Set instance_id if provided in args/kwargs
        if 'instance_id' in kwargs:
            self.instance_id = kwargs['instance_id']
        elif len(args) > 6:
            self.instance_id = args[6]

    def setParent(self, parent):
        pass
    def set_selected(self, selected):
        pass
    def update_legal_actions(self, actions):
        pass

mock_card_widget_module.CardWidget = MockCardWidget
sys.modules["dm_toolkit.gui.widgets.card_widget"] = mock_card_widget_module

# Import after mocking
from dm_toolkit.gui.widgets.zone_widget import ZoneWidget

class TestZoneDisplay(unittest.TestCase):
    """
    Tests for ZoneWidget display logic, focusing on bundled visualization
    and popup interaction for Mana and Graveyard.
    """

    def test_mana_zone_collapsed_by_default(self):
        # "P0 Mana" should trigger is_mana
        widget = ZoneWidget("P0 Mana")

        widget.card_layout = MagicMock()
        widget.card_layout.count.return_value = 0

        card_data = [{'id': 1, 'instance_id': 100}, {'id': 2, 'instance_id': 101}]
        card_db = {1: MagicMock(name="Test Card"), 2: MagicMock(name="Test Card 2")}

        widget.update_cards(card_data, card_db)

        # Should be bundled into 1 widget
        self.assertEqual(len(widget.cards), 1)
        self.assertTrue("Mana" in widget.title)

    def test_graveyard_collapsed_by_default(self):
        # "P1 Graveyard" should trigger is_grave
        widget = ZoneWidget("P1 Graveyard")

        widget.card_layout = MagicMock()
        widget.card_layout.count.return_value = 0

        card_data = [{'id': 1, 'instance_id': 100}]
        card_db = {1: MagicMock(name="Test Card")}

        widget.update_cards(card_data, card_db)
        self.assertEqual(len(widget.cards), 1)

    def test_battle_zone_expanded_by_default(self):
        widget = ZoneWidget("P0 Battle Zone")
        widget.card_layout = MagicMock()
        widget.card_layout.count.return_value = 0

        card_data = [{'id': 1, 'instance_id': 100}, {'id': 2, 'instance_id': 101}]
        card_db = {1: MagicMock(name="Test Card"), 2: MagicMock(name="Test Card 2")}

        widget.update_cards(card_data, card_db)
        # Should NOT be bundled
        self.assertEqual(len(widget.cards), 2)

    def test_popup_connection(self):
        # Verify that clicking the bundled mana card calls _open_popup
        widget = ZoneWidget("P0 Mana")
        widget.card_layout = MagicMock()
        widget.card_layout.count.return_value = 0
        card_data = [{'id': 1, 'instance_id': 100}]
        card_db = {1: MagicMock(name="Test Card")}

        # Patch _open_popup before update_cards so the connection uses the mock
        with patch.object(widget, '_open_popup') as mock_open:
            widget.update_cards(card_data, card_db)

            # Get the card widget
            card_widget = widget.cards[0]

            # Emit click
            card_widget.clicked.emit(0, 0)

            # Since update_cards connects the signal, emitting it should call the mock
            mock_open.assert_called_once()

    def test_expanded_in_popup(self):
        # Verify that if we force collapsed=False (like in popup), it expands
        widget = ZoneWidget("P0 Mana")
        widget.card_layout = MagicMock()
        widget.card_layout.count.return_value = 0
        card_data = [{'id': 1, 'instance_id': 100}, {'id': 2, 'instance_id': 101}]
        card_db = {1: MagicMock(name="Test Card"), 2: MagicMock(name="Test Card 2")}

        widget.update_cards(card_data, card_db, collapsed=False)

        self.assertEqual(len(widget.cards), 2)

if __name__ == '__main__':
    unittest.main()
