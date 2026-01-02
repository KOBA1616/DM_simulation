import unittest
import sys
import os

# Ensure we can import the module
sys.path.append(os.getcwd())

from dm_ai_action_command_shim import translate_action_to_command, Command, MutateCommand, FlowCommand

class MockAction:
    def __init__(self, type_name, **kwargs):
        self.type = type_name
        for k, v in kwargs.items():
            setattr(self, k, v)

class TestCommandShim(unittest.TestCase):
    def test_basic_mappings(self):
        a = MockAction("ADD_MANA", amount=5)
        cmd = translate_action_to_command(a)
        self.assertIsInstance(cmd, MutateCommand)
        self.assertEqual(cmd.payload["add_mana"], 5)

        a = MockAction("DRAW_CARD", count=2)
        cmd = translate_action_to_command(a)
        self.assertIsInstance(cmd, MutateCommand)
        self.assertEqual(cmd.payload["draw"], 2)

    def test_high_priority_mappings(self):
        # BLOCK
        a = MockAction("BLOCK", source_instance_id=10, target_instance_id=20)
        cmd = translate_action_to_command(a)
        self.assertIsInstance(cmd, FlowCommand)
        self.assertEqual(cmd.payload["block"], True)
        self.assertEqual(cmd.payload["blocker"], 10)
        self.assertEqual(cmd.payload["attacker"], 20)

        # BREAK_SHIELD
        a = MockAction("BREAK_SHIELD", source_instance_id=1, target_instance_id=2)
        cmd = translate_action_to_command(a)
        self.assertIsInstance(cmd, MutateCommand)
        self.assertEqual(cmd.payload["break_shield"], True)

    def test_medium_priority_mappings(self):
        # DESTROY
        a = MockAction("DESTROY", target_id=99)
        cmd = translate_action_to_command(a)
        self.assertIsInstance(cmd, MutateCommand)
        self.assertEqual(cmd.payload["destroy"], 99)

        # TAP
        a = MockAction("TAP", instance_id=5)
        cmd = translate_action_to_command(a)
        self.assertIsInstance(cmd, MutateCommand)
        self.assertEqual(cmd.payload["tap"], 5)

    def test_fallback(self):
        a = MockAction("UNKNOWN_TYPE_XYZ")
        cmd = translate_action_to_command(a)
        self.assertEqual(cmd.kind, "generic")
        self.assertEqual(cmd.payload["action_type"], "UNKNOWN_TYPE_XYZ")

    def test_icommand_methods(self):
        # Ensure it has ICommand methods
        a = MockAction("PASS")
        cmd = translate_action_to_command(a)
        self.assertTrue(hasattr(cmd, "execute"))
        self.assertTrue(hasattr(cmd, "invert"))
        self.assertTrue(hasattr(cmd, "to_dict"))

        d = cmd.to_dict()
        self.assertEqual(d["kind"], "flow")
        self.assertEqual(d["payload"]["pass"], True)

if __name__ == "__main__":
    unittest.main()
