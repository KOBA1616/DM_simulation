import unittest
import sys
import os

# Add relevant paths
sys.path.append(os.path.join(os.getcwd(), 'python'))
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    from dm_toolkit.action_mapper import ActionToCommandMapper
except ImportError:
    # If not running in repo structure where dm_toolkit is discoverable
    pass

class TestActionMapper(unittest.TestCase):

    def test_draw_card_mapping(self):
        action = {
            "type": "DRAW_CARD",
            "value1": 2
        }
        cmd = ActionToCommandMapper.map_action(action)
        # Updated Expectation: Specific Command Type Preserved
        self.assertEqual(cmd['type'], "DRAW_CARD")
        self.assertEqual(cmd['from_zone'], "DECK")
        self.assertEqual(cmd['to_zone'], "HAND")
        self.assertEqual(cmd['amount'], 2)

    def test_destroy_mapping(self):
        action = {
            "type": "DESTROY",
            "filter": { "zones": ["BATTLE_ZONE"], "count": 1 }
        }
        cmd = ActionToCommandMapper.map_action(action)
        # Updated Expectation: Specific Command Type Preserved
        self.assertEqual(cmd['type'], "DESTROY")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")
        # from_zone should be unspecified to allow C++ deduction
        self.assertNotIn('from_zone', cmd)

    def test_discard_mapping(self):
        action = {
            "type": "DISCARD",
            "filter": { "zones": ["HAND"], "count": 1 }
        }
        cmd = ActionToCommandMapper.map_action(action)
        # Updated Expectation: Specific Command Type Preserved
        self.assertEqual(cmd['type'], "DISCARD")
        self.assertEqual(cmd['from_zone'], "HAND")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")

    def test_mana_charge_mapping(self):
        action = {
            "type": "MANA_CHARGE",
            "value1": 1
        }
        cmd = ActionToCommandMapper.map_action(action)
        # Updated Expectation: Specific Command Type Preserved
        self.assertEqual(cmd['type'], "MANA_CHARGE")
        self.assertEqual(cmd['to_zone'], "MANA")
        self.assertEqual(cmd['from_zone'], "DECK")

    def test_return_to_hand_mapping(self):
        action = {
            "type": "RETURN_TO_HAND",
            "filter": { "zones": ["BATTLE_ZONE"], "count": 1 }
        }
        cmd = ActionToCommandMapper.map_action(action)
        # Updated Expectation: Specific Command Type Preserved
        self.assertEqual(cmd['type'], "RETURN_TO_HAND")
        self.assertEqual(cmd['to_zone'], "HAND")
        # from_zone usually unspecified for return to hand unless implicit

    def test_send_to_deck_bottom(self):
        action = {
            "type": "SEND_TO_DECK_BOTTOM",
            "value1": 1
        }
        cmd = ActionToCommandMapper.map_action(action)
        # Updated Expectation: Specific Command Type Preserved (Validation will now pass)
        self.assertEqual(cmd['type'], "SEND_TO_DECK_BOTTOM")
        self.assertEqual(cmd['to_zone'], "DECK_BOTTOM")

if __name__ == '__main__':
    unittest.main()
