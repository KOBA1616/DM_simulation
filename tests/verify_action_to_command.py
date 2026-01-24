
import unittest
import sys
import os

# Ensure we can import dm_toolkit
sys.path.append(os.getcwd())

from dm_toolkit.action_to_command import map_action

class TestActionToCommand(unittest.TestCase):

    def test_basic_mana_charge(self):
        action = {"type": "MANA_CHARGE", "source_instance_id": 10, "from_zone": "HAND"}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "MANA_CHARGE")
        self.assertEqual(cmd['instance_id'], 10)
        self.assertEqual(cmd['from_zone'], "HAND")
        self.assertEqual(cmd['to_zone'], "MANA")

    def test_play_card(self):
        action = {"type": "PLAY_CARD", "value1": 5, "source_instance_id": 20}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "PLAY_FROM_ZONE")
        self.assertEqual(cmd['from_zone'], "HAND")
        self.assertEqual(cmd['to_zone'], "BATTLE")
        self.assertEqual(cmd['amount'], 5)
        self.assertEqual(cmd['instance_id'], 20)

    def test_attack_player_with_zero_id(self):
        # source_instance_id is 0. This is a valid ID.
        action = {"type": "ATTACK_PLAYER", "source_instance_id": 0, "target_player": 1}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "ATTACK_PLAYER")
        self.assertEqual(cmd['instance_id'], 0)
        self.assertEqual(cmd['target_player'], 1)

    def test_attack_creature_with_zero_id(self):
        # attacker 0 attacks target 5
        action = {"type": "ATTACK_CREATURE", "source_instance_id": 0, "target_instance_id": 5}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "ATTACK_CREATURE")
        self.assertEqual(cmd['instance_id'], 0)
        self.assertEqual(cmd['target_instance'], 5)

        # attacker 5 attacks target 0
        action = {"type": "ATTACK_CREATURE", "source_instance_id": 5, "target_instance_id": 0}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "ATTACK_CREATURE")
        self.assertEqual(cmd['instance_id'], 5)
        self.assertEqual(cmd['target_instance'], 0)

    def test_value1_to_amount_zero(self):
        # value1 is 0. amount should be 0.
        action = {"type": "TEST_ACTION", "value1": 0}
        cmd = map_action(action)
        self.assertEqual(cmd['amount'], 0)

    def test_options_recursion(self):
        action = {
            "type": "CHOICE",
            "options": [
                {"type": "MANA_CHARGE", "source_instance_id": 1},
                {"type": "MANA_CHARGE", "source_instance_id": 2}
            ]
        }
        cmd = map_action(action)
        self.assertTrue('options' in cmd)
        self.assertEqual(len(cmd['options']), 2)
        # options are wrapped in list
        self.assertEqual(cmd['options'][0][0]['type'], "MANA_CHARGE")
        self.assertEqual(cmd['options'][0][0]['instance_id'], 1)

    def test_block_creature_with_zero_id(self):
         # blocker 0 blocks 1
        action = {"type": "BLOCK_CREATURE", "source_instance_id": 0, "target_instance_id": 1}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "FLOW")
        self.assertEqual(cmd['flow_type'], "BLOCK")
        self.assertEqual(cmd['instance_id'], 0)
        self.assertEqual(cmd['target_instance'], 1)

    def test_zero_value_implies_all(self):
        """Verify that value1: 0 implies amount: 255 (ALL) for specific commands."""
        target_types = ["DESTROY", "DISCARD", "RETURN_TO_HAND", "TAP", "UNTAP", "SEND_TO_MANA"]

        for t in target_types:
            action = {"type": t, "value1": 0}
            cmd = map_action(action)
            self.assertEqual(cmd['amount'], 255, f"{t} with value1=0 should map to amount=255")

        # Test generic MOVE_CARD (maps to TRANSITION or specific)
        action = {"type": "MOVE_CARD", "value1": 0, "to_zone": "GRAVEYARD"}
        cmd = map_action(action)
        # Assuming maps to TRANSITION or DESTROY depending on inferrence, but handle move card logic applies 255
        self.assertEqual(cmd['amount'], 255, "MOVE_CARD with value1=0 should map to amount=255")

if __name__ == '__main__':
    unittest.main()
