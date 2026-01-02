
import unittest
import uuid
from dm_toolkit.action_mapper import ActionToCommandMapper

class TestActionMapper(unittest.TestCase):

    def test_basic_move(self):
        action = {
            "type": "MOVE_CARD",
            "from_zone": "HAND",
            "to_zone": "MANA_ZONE",
            "card_id": 10
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['type'], "MANA_CHARGE")
        self.assertEqual(cmd['from_zone'], "HAND")
        self.assertEqual(cmd['to_zone'], "MANA_ZONE")
        self.assertTrue(cmd['uid'])

    def test_options_recursion(self):
        action = {
            "type": "SELECT_OPTION",
            "options": [
                [{"type": "draw_card", "value1": 1}],
                [{"type": "mana_charge", "value1": 1}]
            ]
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['type'], "CHOICE")
        self.assertEqual(len(cmd['options']), 2)
        # Check first option
        opt1 = cmd['options'][0][0]
        self.assertEqual(opt1['type'], "TRANSITION") # Draw maps to transition
        self.assertEqual(opt1['amount'], 1)

    def test_numeric_normalization(self):
        action = {
            "type": "POWER_MOD",
            "str_val": "POWER_MOD",
            "value1": 5000
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['type'], "POWER_MOD")
        self.assertEqual(cmd['amount'], 5000)

    def test_filter_copy(self):
        f = {"races": ["Dragon"]}
        action = {
            "type": "SEARCH_DECK",
            "filter": f
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['target_filter'], f)
        self.assertIsNot(cmd['target_filter'], f) # Should be a copy

if __name__ == '__main__':
    unittest.main()
