
import unittest
import sys
import os

# Ensure we can import dm_toolkit and bin
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
    CommandType = dm_ai_module.CommandType
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    CommandType = None

from dm_toolkit.action_to_command import map_action
from dm_toolkit import command_builders

class TestBufferActions(unittest.TestCase):

    def test_look_to_buffer(self):
        # Migrated to builder
        if HAS_NATIVE:
            cmd_def = command_builders.build_look_to_buffer(
                look_count=3,
                from_zone="SHIELD",
                native=True
            )
            cmd = map_action(cmd_def)
            self.assertEqual(cmd['type'], CommandType.LOOK_TO_BUFFER)
        else:
            # Fallback for non-native env (though we are in native env)
            action = command_builders.build_look_to_buffer(3, from_zone="SHIELD", native=False)
            cmd = map_action(action)
            self.assertEqual(cmd['type'], "LOOK_TO_BUFFER")

        self.assertEqual(cmd.get('amount'), 3)
        self.assertEqual(cmd.get('from_zone'), "SHIELD")

    def test_reveal_to_buffer(self):
        if HAS_NATIVE:
            cmd_def = command_builders.build_reveal_to_buffer(2, from_zone="DECK", native=True)
            cmd = map_action(cmd_def)
            self.assertEqual(cmd['type'], CommandType.REVEAL_TO_BUFFER)
        else:
            action = command_builders.build_reveal_to_buffer(2, from_zone="DECK", native=False)
            cmd = map_action(action)
            self.assertEqual(cmd['type'], "REVEAL_TO_BUFFER")

        self.assertEqual(cmd.get('amount'), 2)
        self.assertEqual(cmd.get('from_zone'), "DECK")

    def test_select_from_buffer(self):
        if HAS_NATIVE:
            cmd_def = command_builders.build_select_from_buffer(1, native=True)
            cmd = map_action(cmd_def)
            self.assertEqual(cmd['type'], CommandType.SELECT_FROM_BUFFER)
        else:
            action = command_builders.build_select_from_buffer(1, native=False)
            cmd = map_action(action)
            self.assertEqual(cmd['type'], "SELECT_FROM_BUFFER")

        self.assertEqual(cmd['amount'], 1)

    def test_play_from_buffer(self):
        if HAS_NATIVE:
            cmd_def = command_builders.build_play_from_buffer(5, to_zone="BATTLE", native=True)
            cmd = map_action(cmd_def)
            self.assertEqual(cmd['type'], CommandType.PLAY_FROM_BUFFER)
            # Native CommandDef to_dict uses 'to_zone'
            self.assertEqual(cmd.get('to_zone'), "BATTLE")
            self.assertEqual(cmd.get('from_zone'), "BUFFER")
            # amount maps to max_cost in builder?
            # build_play_from_buffer(native=True) uses amount=max_cost
            self.assertEqual(cmd.get('amount'), 5)
        else:
            action = command_builders.build_play_from_buffer(5, to_zone="BATTLE_ZONE", native=False)
            cmd = map_action(action)
            self.assertEqual(cmd['type'], "PLAY_FROM_BUFFER")
            self.assertEqual(cmd['from_zone'], "BUFFER")
            self.assertEqual(cmd['to_zone'], "BATTLE") # Normalized
            self.assertEqual(cmd['max_cost'], 5)

    def test_move_buffer_to_zone(self):
        if HAS_NATIVE:
            cmd_def = command_builders.build_move_buffer_to_zone(1, to_zone="HAND", native=True)
            cmd = map_action(cmd_def)
            # Native builder maps this to TRANSITION? No, _build_native_command maps MOVE_BUFFER_TO_ZONE if it exists.
            # Does CommandType have MOVE_BUFFER_TO_ZONE? Yes.
            # Wait, build_move_buffer_to_zone calls _build_native_command("MOVE_BUFFER_TO_ZONE", ...)
            self.assertEqual(cmd['type'], CommandType.MOVE_BUFFER_TO_ZONE)
            self.assertEqual(cmd['from_zone'], "BUFFER")
            self.assertEqual(cmd['to_zone'], "HAND")
            self.assertEqual(cmd['amount'], 1)
        else:
            action = command_builders.build_move_buffer_to_zone(1, to_zone="HAND", native=False)
            cmd = map_action(action)
            # Legacy mapping logic maps it to TRANSITION
            self.assertEqual(cmd['type'], "TRANSITION")
            self.assertEqual(cmd['from_zone'], "BUFFER")
            self.assertEqual(cmd['to_zone'], "HAND")
            self.assertEqual(cmd['amount'], 1)

if __name__ == '__main__':
    unittest.main()
