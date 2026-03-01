import sys
import unittest
import importlib
from unittest.mock import MagicMock, patch
from typing import Protocol, runtime_checkable

class TestCommandsConsistency(unittest.TestCase):
    def setUp(self):
        # Create a mock for dm_ai_module
        self.mock_dm_ai_module = MagicMock()
        # Mock ActionGenerator
        self.mock_dm_ai_module.ActionGenerator = MagicMock()

        # Patch sys.modules to return the mock for dm_ai_module
        self.modules_patcher = patch.dict(sys.modules, {'dm_ai_module': self.mock_dm_ai_module})
        self.modules_patcher.start()

        # Import the module under test after patching
        import dm_toolkit.commands
        self.commands_module = dm_toolkit.commands
        # Reload to ensure it picks up the patched dm_ai_module
        importlib.reload(self.commands_module)

    def tearDown(self):
        self.modules_patcher.stop()

    def test_icommand_protocol(self):
        """Check ICommand interface is a Protocol and runtime checkable"""
        ICommand = self.commands_module.ICommand
        self.assertTrue(issubclass(ICommand, Protocol), "ICommand should be a Protocol")

        # Define a class that implements the protocol
        class MyCmd:
            def execute(self, state, card_db=None): pass
            def invert(self, state): pass
            def to_dict(self): return {}

        # Verify isinstance works
        self.assertTrue(isinstance(MyCmd(), ICommand))

        # Define a class that DOES NOT implement the protocol
        class NotCmd:
            pass

        self.assertFalse(isinstance(NotCmd(), ICommand))

    def test_wrap_action_already_command(self):
        """wrap_action should return object as is if it has execute method"""
        class MyCmd:
            def execute(self, state, card_db=None): pass
            # Note: wrap_action currently only checks for 'execute'.
            # It casts to ICommand but does not strictly check other methods at runtime inside the function.
            # But let's verify it returns the same object.

        cmd = MyCmd()
        wrapped = self.commands_module.wrap_action(cmd)
        self.assertIs(wrapped, cmd)

    @patch('dm_toolkit.engine.compat.EngineCompat.ExecuteCommand')
    def test_wrap_action_execution(self, mock_execute):
        """wrapped action should call EngineCompat.ExecuteCommand when executed"""
        action = {"type": "PASS"}
        wrapped = self.commands_module.wrap_action(action)

        # Verify wrapped object is an ICommand
        self.assertTrue(isinstance(wrapped, self.commands_module.ICommand))

        state = MagicMock()
        card_db = MagicMock()

        # Execute
        wrapped.execute(state, card_db)

        # Verify EngineCompat.ExecuteCommand was called
        mock_execute.assert_called()
        args, _ = mock_execute.call_args
        self.assertEqual(args[0], state)
        # The second arg should be a command dict.
        # wrap_action uses ensure_executable_command which calls map_action.
        # map_action maps "PASS" to "PASS".
        self.assertEqual(args[1]['type'], 'PASS')

        # Verify to_dict works
        d = wrapped.to_dict()
        self.assertEqual(d['type'], 'PASS')

    def test_generate_legal_commands(self):
        """generate_legal_commands calls ActionGenerator and wraps results"""
        # Setup mock return for generate_legal_actions
        action1 = {"type": "PASS"}
        action2 = {"type": "MANA_CHARGE", "value1": 1}

        # The generate_legal_commands function calls dm_ai_module.ActionGenerator.generate_legal_actions
        self.mock_dm_ai_module.ActionGenerator.generate_legal_actions.return_value = [action1, action2]

        state = MagicMock()
        card_db = MagicMock()

        cmds = self.commands_module.generate_legal_commands(state, card_db)

        # Check call to generator
        self.mock_dm_ai_module.ActionGenerator.generate_legal_actions.assert_called_with(state, card_db)

        # Check results
        self.assertEqual(len(cmds), 2)

        # Verify they are wrapped commands
        self.assertTrue(isinstance(cmds[0], self.commands_module.ICommand))
        self.assertTrue(isinstance(cmds[1], self.commands_module.ICommand))

        # Verify content via to_dict
        self.assertEqual(cmds[0].to_dict()['type'], 'PASS')
        self.assertEqual(cmds[1].to_dict()['type'], 'MANA_CHARGE')
        self.assertEqual(cmds[1].to_dict()['amount'], 1)  # map_action maps value1 to amount

    def test_generate_legal_commands_empty(self):
        """generate_legal_commands handles empty or None return from generator"""
        self.mock_dm_ai_module.ActionGenerator.generate_legal_actions.return_value = None

        state = MagicMock()
        card_db = MagicMock()

        cmds = self.commands_module.generate_legal_commands(state, card_db)
        self.assertEqual(cmds, [])

if __name__ == '__main__':
    unittest.main()
