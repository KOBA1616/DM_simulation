
import unittest
import uuid
from dm_toolkit.action_to_command import map_action

class TestActionMapper(unittest.TestCase):

    def test_basic_move(self):
        action = {
            "type": "MOVE_CARD",
            "from_zone": "HAND",
            "to_zone": "MANA_ZONE",
            "card_id": 10
        }
        cmd = map_action(action)
        # In the new system, generic moves are TRANSITIONs
        self.assertEqual(cmd['type'], "TRANSITION")
        self.assertEqual(cmd['from_zone'], "HAND")
        self.assertEqual(cmd['to_zone'], "MANA") # Zone names are normalized
        self.assertTrue(cmd['uid'])

    def test_options_recursion(self):
        action = {
            "type": "SELECT_OPTION",
            "options": [
                [{"type": "draw_card", "value1": 1}],
                [{"type": "mana_charge", "value1": 1}]
            ]
        }
        cmd = map_action(action)
        # SELECT_OPTION maps to QUERY for player choice
        self.assertEqual(cmd['type'], "QUERY")
        self.assertEqual(len(cmd['options']), 2)
        # Check first option
        opt1 = cmd['options'][0][0]
        # DRAW_CARD maps to DRAW_CARD (not TRANSITION)
        self.assertEqual(opt1['type'], "DRAW_CARD")
        self.assertEqual(opt1['amount'], 1)

    def test_numeric_normalization(self):
        action = {
            "type": "POWER_MOD",
            "str_val": "POWER_MOD",
            "value1": 5000
        }
        cmd = map_action(action)
        # POWER_MOD is a type of MUTATION in the new system
        self.assertEqual(cmd['type'], "MUTATE")
        self.assertEqual(cmd['mutation_kind'], "POWER_MOD")
        self.assertEqual(cmd['amount'], 5000)

    def test_filter_copy(self):
        f = {"races": ["Dragon"]}
        action = {
            "type": "SEARCH_DECK",
            "filter": f
        }
        cmd = map_action(action)
        self.assertEqual(cmd['target_filter'], f)
        self.assertIsNot(cmd['target_filter'], f) # Should be a copy

if __name__ == '__main__':
    unittest.main()
