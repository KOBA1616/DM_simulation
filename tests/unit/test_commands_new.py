
import unittest
from typing import Any, Dict, Optional
from dm_toolkit.commands_new import wrap_action, ICommand

class MockAction:
    def __init__(self, type_name="MOCK"):
        self.type = type_name
        self.value1 = 42

    def to_dict(self):
        return {"type": self.type, "value1": self.value1}

class MockCommand:
    def execute(self, state: Any) -> Optional[Any]:
        return True

    def invert(self, state: Any) -> Optional[Any]:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "COMMAND"}

class TestCommandsNew(unittest.TestCase):

    def test_wrap_legacy_action(self):
        act = MockAction("DRAW_CARD")
        cmd = wrap_action(act)
        self.assertTrue(isinstance(cmd, ICommand))
        d = cmd.to_dict()
        self.assertEqual(d['type'], "TRANSITION") # DRAW maps to TRANSITION
        self.assertEqual(d['amount'], 42)

    def test_wrap_existing_command(self):
        base_cmd = MockCommand()
        wrapped = wrap_action(base_cmd)
        self.assertEqual(wrapped, base_cmd)
        self.assertEqual(wrapped.to_dict()['type'], "COMMAND")

    def test_wrap_dict(self):
        act_dict = {"type": "TAP", "source_zone": "BATTLE_ZONE"}
        cmd = wrap_action(act_dict)
        d = cmd.to_dict()
        self.assertEqual(d['type'], "TAP")

if __name__ == '__main__':
    unittest.main()
