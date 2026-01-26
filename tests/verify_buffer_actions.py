
import unittest
import sys
import os

# Ensure we can import dm_toolkit
sys.path.append(os.getcwd())

from dm_toolkit.action_to_command import map_action

class TestBufferActions(unittest.TestCase):

    def test_look_to_buffer(self):
        # Action with specific zone
        action = {"type": "LOOK_TO_BUFFER", "value1": 3, "from_zone": "SHIELD"}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "LOOK_TO_BUFFER")
        self.assertEqual(cmd['look_count'], 3)
        # Verify from_zone is preserved (Legacy issue check)
        self.assertEqual(cmd.get('from_zone'), "SHIELD", "from_zone should be preserved for LOOK_TO_BUFFER")
        # Verify amount is inferred from value1 via finalize
        self.assertEqual(cmd.get('amount'), 3)

    def test_reveal_to_buffer(self):
        # Action with default zone implied
        action = {"type": "REVEAL_TO_BUFFER", "value1": 2, "from_zone": "DECK"}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "REVEAL_TO_BUFFER")
        self.assertEqual(cmd['look_count'], 2)
        self.assertEqual(cmd.get('from_zone'), "DECK", "from_zone should be preserved for REVEAL_TO_BUFFER")
        self.assertEqual(cmd.get('amount'), 2)

    def test_select_from_buffer(self):
        action = {"type": "SELECT_FROM_BUFFER", "value1": 1}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "SELECT_FROM_BUFFER")
        self.assertEqual(cmd['amount'], 1)

    def test_play_from_buffer(self):
        action = {"type": "PLAY_FROM_BUFFER", "value1": 5, "to_zone": "BATTLE_ZONE"}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "PLAY_FROM_BUFFER")
        self.assertEqual(cmd['from_zone'], "BUFFER")
        self.assertEqual(cmd['to_zone'], "BATTLE") # Normalized to BATTLE
        self.assertEqual(cmd['max_cost'], 5)

    def test_move_buffer_to_zone(self):
        action = {"type": "MOVE_BUFFER_TO_ZONE", "value1": 1, "to_zone": "HAND"}
        cmd = map_action(action)
        self.assertEqual(cmd['type'], "TRANSITION")
        self.assertEqual(cmd['from_zone'], "BUFFER")
        self.assertEqual(cmd['to_zone'], "HAND")
        self.assertEqual(cmd['amount'], 1)

if __name__ == '__main__':
    unittest.main()
