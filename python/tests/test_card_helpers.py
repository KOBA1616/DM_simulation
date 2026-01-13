
import sys
import unittest
from unittest.mock import MagicMock, Mock

# Mock dm_ai_module and other C++ dependencies
sys.modules['dm_ai_module'] = MagicMock()

# Mock PyQt6 to run without GUI
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()
sys.modules['PyQt6.QtGui'] = MagicMock()

from dm_toolkit.gui.utils.card_helpers import convert_card_data_to_dict

class MockCivilization:
    def __init__(self, name):
        self.name = name

class MockCardType:
    def __init__(self, name):
        self.name = name

class MockEffect:
    def __init__(self, trigger, commands=None):
        self.trigger = Mock(name=trigger)
        self.trigger.name = trigger
        self.commands = commands or []
        self.condition = None
        self.actions = []

class MockCommand:
    def __init__(self, ctype, amount=0):
        self.type = Mock(name=ctype)
        self.type.name = ctype
        self.amount = amount
        self.target_group = None
        self.target_filter = None

class MockCardData:
    def __init__(self):
        self.id = 1
        self.name = "Test Card"
        self.cost = 5
        self.power = 5000
        self.civilizations = [MockCivilization("FIRE")]
        self.races = ["Dragon"]
        self.type = MockCardType("CREATURE")

        # Use a MagicMock for keywords so hasattr works but we can control specific attributes
        self.keywords = MagicMock()
        self.keywords.speed_attacker = True
        self.keywords.blocker = False

        self.effects = [
            MockEffect("ON_PLAY", [MockCommand("DRAW_CARD", 1)])
        ]
        self.spell_side = None
        self.static_abilities = []
        self.revolution_change_condition = None
        self.is_key_card = False
        self.ai_importance_score = 0

class TestCardHelpers(unittest.TestCase):
    def test_convert_card_data_to_dict(self):
        card = MockCardData()
        data = convert_card_data_to_dict(card)

        self.assertEqual(data['id'], 1)
        self.assertEqual(data['name'], "Test Card")
        self.assertEqual(data['cost'], 5)
        self.assertEqual(data['power'], 5000)
        self.assertEqual(data['civilizations'], ["FIRE"])
        self.assertEqual(data['races'], ["Dragon"])
        self.assertEqual(data['type'], "CREATURE")

        self.assertTrue(data['keywords']['speed_attacker'])
        self.assertFalse(data['keywords']['blocker'])

        # Check effects
        self.assertEqual(len(data['effects']), 1)
        self.assertEqual(data['effects'][0]['trigger'], "ON_PLAY")
        self.assertEqual(data['effects'][0]['commands'][0]['type'], "DRAW_CARD")
        self.assertEqual(data['effects'][0]['commands'][0]['amount'], 1)

if __name__ == '__main__':
    unittest.main()
