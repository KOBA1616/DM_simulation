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

    def setUp(self):
        # Ensure we can import even if pathing is tricky in test env
        if 'dm_toolkit.action_mapper' not in sys.modules:
            try:
                from dm_toolkit.action_mapper import ActionToCommandMapper
                global ActionToCommandMapper
            except ImportError:
                self.skipTest("dm_toolkit.action_mapper not found")

    def test_draw_card_mapping(self):
        action = {
            "type": "DRAW_CARD",
            "value1": 2
        }
        cmd = ActionToCommandMapper.map_action(action)

        # New implementation prefers DRAW_CARD macro type
        if cmd['type'] == "DRAW_CARD":
            self.assertEqual(cmd['type'], "DRAW_CARD")
            self.assertEqual(cmd.get('amount'), 2)
        else:
            # Fallback to TRANSITION
            self.assertEqual(cmd['type'], "TRANSITION")
            self.assertEqual(cmd['from_zone'], "DECK")
            self.assertEqual(cmd['to_zone'], "HAND")
            self.assertEqual(cmd['amount'], 2)

    def test_destroy_mapping(self):
        action = {
            "type": "DESTROY",
            "filter": { "zones": ["BATTLE_ZONE"], "count": 1 }
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['type'], "DESTROY")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")

    def test_discard_mapping(self):
        action = {
            "type": "DISCARD",
            "filter": { "zones": ["HAND"], "count": 1 }
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['type'], "DISCARD")
        self.assertEqual(cmd['to_zone'], "GRAVEYARD")
        self.assertEqual(cmd['from_zone'], "HAND")

    def test_mana_charge_mapping(self):
        action = {
            "type": "MANA_CHARGE",
            "value1": 1
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['type'], "MANA_CHARGE")
        self.assertEqual(cmd['to_zone'], "MANA_ZONE")

    def test_return_to_hand_mapping(self):
        action = {
            "type": "RETURN_TO_HAND",
            "filter": { "zones": ["BATTLE_ZONE"], "count": 1 }
        }
        cmd = ActionToCommandMapper.map_action(action)
        self.assertEqual(cmd['type'], "RETURN_TO_HAND")
        self.assertEqual(cmd['to_zone'], "HAND")

    def test_send_to_deck_bottom(self):
        action = {
            "type": "SEND_TO_DECK_BOTTOM",
            "value1": 1
        }
        cmd = ActionToCommandMapper.map_action(action)
        # Implementation reverted to TRANSITION -> DECK_BOTTOM to match legacy behavior
        self.assertEqual(cmd['type'], "TRANSITION")
        self.assertEqual(cmd['to_zone'], "DECK_BOTTOM")

if __name__ == '__main__':
    unittest.main()
