
import unittest
import sys
import os
from enum import Enum

# Add root to path for dm_toolkit
sys.path.append(os.getcwd())
# Add bin for dm_ai_module
sys.path.append(os.path.join(os.getcwd(), 'bin'))

# Mock dm_ai_module if missing
try:
    import dm_ai_module
except ImportError:
    from unittest.mock import MagicMock
    dm_ai_module = MagicMock()
    sys.modules['dm_ai_module'] = dm_ai_module

import dm_toolkit.action_to_command as module_under_test
from dm_toolkit.action_to_command import map_action, set_command_type_enum

# Define a Mock Enum that includes the new type
class MockCommandType(Enum):
    PLAY_FROM_ZONE = 100
    TRANSITION = 1
    NONE = 17

class TestPlayCardMapping(unittest.TestCase):
    def setUp(self):
        # Save and restore global enum injection to avoid cross-test contamination.
        self._original_command_type = getattr(module_under_test, '_CommandType', None)
        # Inject our mock enum
        set_command_type_enum(MockCommandType)

    def tearDown(self):
        set_command_type_enum(self._original_command_type)

    def test_play_card_mapping(self):
        # Action data representing PLAY_CARD
        action = {
            'type': 'PLAY_CARD',
            'card_id': 1,
            'source_instance_id': 100,
            'target_player': 0
        }

        cmd = map_action(action)

        print(f"Mapped Command: {cmd}")

        self.assertEqual(cmd['type'], 'PLAY_FROM_ZONE')
        self.assertEqual(cmd['from_zone'], 'HAND')
        self.assertEqual(cmd['to_zone'], 'BATTLE')
        self.assertEqual(cmd['instance_id'], 100)
        self.assertEqual(cmd['legacy_original_type'], 'PLAY_CARD')

    def test_declare_play_mapping(self):
        # Action data representing DECLARE_PLAY (another play variant)
        action = {
            'type': 'DECLARE_PLAY',
            'card_id': 1,
            'source_instance_id': 101,
            'target_player': 0
        }

        cmd = map_action(action)

        self.assertEqual(cmd['type'], 'PLAY_FROM_ZONE')
        self.assertEqual(cmd['instance_id'], 101)

if __name__ == '__main__':
    unittest.main()
