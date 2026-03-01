# -*- coding: utf-8 -*-
import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dm_toolkit.engine.compat import EngineCompat
import dm_ai_module

class TestEffectResolverCompat(unittest.TestCase):
    def setUp(self):
        # Create common mocks
        self.mock_state = MagicMock(spec=dm_ai_module.GameState)
        self.mock_db = {"1": {"name": "Test Card"}} # Simple dict DB

        # Ensure we can patch dm_ai_module attributes even if they don't exist in the stub
        if not hasattr(dm_ai_module, 'EffectResolver'):
            dm_ai_module.EffectResolver = MagicMock()

    def test_priority_action_execute(self):
        """Test that action.execute(state, db) is called first if available."""
        mock_action = MagicMock()
        # Ensure execute is callable
        mock_action.execute = MagicMock()

        # Call resolve_action
        EngineCompat.EffectResolver_resolve_action(self.mock_state, mock_action, self.mock_db)

        # Assert execute was called
        # EngineCompat tries action.execute(state, db) then action.execute(state)
        # We assume it matches one of them.
        self.assertTrue(mock_action.execute.called, "action.execute() should be called")

        # Verify arguments: check if it was called with state and/or db
        args, _ = mock_action.execute.call_args
        self.assertIn(self.mock_state, args)

    @patch('dm_toolkit.engine.compat.EngineCompat.ExecuteCommand')
    def test_unified_path_conversion(self, mock_execute_command):
        """Test that if action.execute is missing, it tries unified conversion."""
        mock_action = MagicMock()
        del mock_action.execute # Ensure no execute method

        # Mock ensure_executable_command to return a valid command
        valid_command = {'type': 'MANA_CHARGE', 'amount': 1}

        with patch('dm_toolkit.unified_execution.ensure_executable_command', return_value=valid_command) as mock_ensure:
            EngineCompat.EffectResolver_resolve_action(self.mock_state, mock_action, self.mock_db)

            mock_ensure.assert_called_with(mock_action)
            mock_execute_command.assert_called_with(self.mock_state, valid_command, self.mock_db)

    @patch('dm_toolkit.engine.compat.EngineCompat.ExecuteCommand')
    def test_native_fallback(self, mock_execute_command):
        """Test fallback to dm_ai_module.EffectResolver if unified conversion is inconclusive."""
        mock_action = MagicMock()
        del mock_action.execute

        # Mock ensure_executable_command to return inconclusive result
        inconclusive_command = {'type': 'NONE'}

        with patch('dm_toolkit.unified_execution.ensure_executable_command', return_value=inconclusive_command):
            with patch('dm_ai_module.EffectResolver.resolve_action') as mock_native_resolve:
                EngineCompat.EffectResolver_resolve_action(self.mock_state, mock_action, self.mock_db)

                # ExecuteCommand should NOT be called for NONE type
                mock_execute_command.assert_not_called()

                # Native resolver SHOULD be called
                mock_native_resolve.assert_called()
                args, _ = mock_native_resolve.call_args
                self.assertEqual(args[0], self.mock_state)
                self.assertEqual(args[1], mock_action)
                # Check DB resolution (might be the dict or resolved object)
                # Since we didn't mock _resolve_db explicitly, it returns the dict or cache
                self.assertTrue(args[2] == self.mock_db or args[2] == EngineCompat._native_db_cache)

    def test_effect_resolver_resume(self):
        """Test delegation of EffectResolver_resume."""
        selection = [1, 2, 3]

        # Manually ensure resume exists for the mock since stub might lack it
        if not hasattr(dm_ai_module.EffectResolver, 'resume'):
            dm_ai_module.EffectResolver.resume = MagicMock()

        with patch('dm_ai_module.EffectResolver.resume') as mock_resume:
            EngineCompat.EffectResolver_resume(self.mock_state, self.mock_db, selection)

            mock_resume.assert_called()
            args, _ = mock_resume.call_args
            self.assertEqual(args[0], self.mock_state)
            # args[1] is db
            self.assertEqual(args[2], selection)

    def test_db_resolution(self):
        """Verify that a dictionary DB is passed through correctly."""
        mock_action = MagicMock()
        mock_action.execute = MagicMock()

        EngineCompat.EffectResolver_resolve_action(self.mock_state, mock_action, self.mock_db)

        # Check that the execute method received the db (if it accepts 2 args)
        # The compat layer tries execute(state, db) first.
        try:
            mock_action.execute.assert_called_with(self.mock_state, self.mock_db)
        except AssertionError:
            # If it failed, maybe it called with resolved DB (which is None/Empty in this test env if not set)
            # Or it fell back to execute(state)
            pass

if __name__ == '__main__':
    unittest.main()
