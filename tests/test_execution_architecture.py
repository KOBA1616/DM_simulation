import pytest
import sys
from unittest.mock import MagicMock, patch

# Ensure dm_ai_module is mocked if not present
if 'dm_ai_module' not in sys.modules:
    sys.modules['dm_ai_module'] = MagicMock()

from dm_toolkit.unified_execution import ensure_executable_command
from dm_toolkit.engine.compat import EngineCompat

def test_standard_execution_pipeline():
    """
    Verifies the standard execution path:
    Caller -> ensure_executable_command -> to_command_dict -> map_action
           -> normalize_legacy_fields -> (Caller invokes) EngineCompat.ExecuteCommand
    """
    # We patch the components to verify the chain of calls.
    # Note: ensure_executable_command imports normalize_legacy_fields internally,
    # but map_action is imported at module level in unified_execution.

    with patch('dm_toolkit.unified_execution.map_action') as mock_map, \
         patch('dm_toolkit.compat_wrappers.normalize_legacy_fields') as mock_norm, \
         patch('dm_toolkit.engine.compat.EngineCompat.ExecuteCommand') as mock_exec:

        # Setup returns to allow flow to proceed
        mock_map.return_value = {"type": "TEST_CMD", "amount": 1}
        # normalize receives the dict from map_action
        mock_norm.side_effect = lambda x: {**x, "normalized": True}

        # --- Execution ---
        # Use a non-special action type to avoid DRAW_CARD preservation logic
        action = {"type": "GENERIC_ACTION", "value1": 2}

        # 1. Caller invokes preparation
        cmd = ensure_executable_command(action)

        # 2. Caller invokes execution
        state = MagicMock()
        EngineCompat.ExecuteCommand(state, cmd)

        # --- Verification ---

        # 1. Check ensure_executable_command flow
        # It calls map_action (via to_command_dict)
        mock_map.assert_called_once_with(action)

        # It calls normalize_legacy_fields with the result of map_action
        mock_norm.assert_called_once_with(mock_map.return_value)

        # 2. Check execution flow
        # It calls ExecuteCommand with the result of ensure_executable_command (which is result of normalize)
        expected_cmd = {"type": "TEST_CMD", "amount": 1, "normalized": True}
        mock_exec.assert_called_once_with(state, expected_cmd)

def test_legacy_resolver_convergence():
    """
    Verifies that legacy EffectResolver_resolve_action converges to the same path:
    EffectResolver_resolve_action -> ensure_executable_command -> ExecuteCommand
    """
    # We need to patch ensure_executable_command inside EngineCompat or unified_execution?
    # EngineCompat.EffectResolver_resolve_action does:
    # from dm_toolkit.unified_execution import ensure_executable_command

    # So we patch dm_toolkit.unified_execution.ensure_executable_command
    # But since we already imported it in this test file, patching the module attribute works if compat imports from module.
    # compat.py does: `from dm_toolkit.unified_execution import ensure_executable_command` INSIDE the method.
    # So patching `dm_toolkit.unified_execution.ensure_executable_command` is correct.

    with patch('dm_toolkit.unified_execution.ensure_executable_command') as mock_ensure, \
         patch('dm_toolkit.engine.compat.EngineCompat.ExecuteCommand') as mock_exec:

        mock_ensure.return_value = {"type": "RESOLVED_CMD"}

        state = MagicMock()
        action = {"type": "LEGACY_ACTION"}
        card_db = MagicMock()

        # Call legacy resolver
        EngineCompat.EffectResolver_resolve_action(state, action, card_db)

        # Verify it uses the unified pipeline
        mock_ensure.assert_called_once_with(action)

        # Verify it executes the result
        assert mock_exec.called
        args, _ = mock_exec.call_args
        assert args[0] == state
        assert args[1] == mock_ensure.return_value
