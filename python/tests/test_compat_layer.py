
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dm_ai_module if it's not available in the environment
sys.modules['dm_ai_module'] = MagicMock()
import dm_ai_module

from dm_toolkit.engine.compat import EngineCompat

class TestEngineCompat(unittest.TestCase):

    def test_get_game_state_attribute_defaults(self):
        # Create a mock that raises AttributeError for any attribute access
        # so getattr(state, 'attr', default) returns default.
        class EmptyState:
            pass

        state = EmptyState()

        # Test defaults
        self.assertEqual(EngineCompat.get_turn_number(state), '?')
        self.assertEqual(EngineCompat.get_active_player_id(state), 0)
        self.assertEqual(EngineCompat.get_current_phase(state), 'UNKNOWN')
        self.assertEqual(EngineCompat.get_pending_query(state), None)
        self.assertEqual(EngineCompat.get_effect_buffer(state), [])
        self.assertEqual(EngineCompat.get_command_history(state), [])

    def test_get_game_state_attribute_aliases(self):
        class AliasState:
            pass

        state = AliasState()

        # Test active_player alias
        state.active_player = 1
        self.assertEqual(EngineCompat.get_active_player_id(state), 1)

        # Test phase alias
        state.phase = "MAIN"
        self.assertEqual(EngineCompat.get_current_phase(state), "MAIN")

    def test_get_player_aliases(self):
        state = MagicMock()
        p0 = MagicMock()
        p1 = MagicMock()

        # Modern: players list
        state.players = [p0, p1]
        # Ensure player0/player1 are not accessed or don't interfere if they exist
        state.player0 = None
        state.player1 = None

        self.assertEqual(EngineCompat.get_player(state, 0), p0)
        self.assertEqual(EngineCompat.get_player(state, 1), p1)

        # Legacy: player0/player1
        del state.players
        # MagicMock will return a mock for players if accessed, so we need a cleaner object or explicit None
        class LegacyState:
            def __init__(self):
                self.player0 = p0
                self.player1 = p1
                self.players = None

        state_legacy = LegacyState()
        self.assertEqual(EngineCompat.get_player(state_legacy, 0), p0)
        self.assertEqual(EngineCompat.get_player(state_legacy, 1), p1)

    def test_action_attributes(self):
        class Action:
            pass

        action = Action()
        action.slot_index = 5
        action.source_instance_id = 10

        self.assertEqual(EngineCompat.get_action_slot_index(action), 5)
        self.assertEqual(EngineCompat.get_action_source_id(action), 10)

        # Test default
        del action.slot_index
        del action.source_instance_id
        self.assertEqual(EngineCompat.get_action_slot_index(action), -1)
        self.assertEqual(EngineCompat.get_action_source_id(action), -1)

    @patch('dm_ai_module.EffectResolver')
    def test_api_wrappers(self, mock_effect_resolver):
        state = MagicMock()
        card_db = {}
        action = MagicMock()

        EngineCompat.EffectResolver_resolve_action(state, action, card_db)
        dm_ai_module.EffectResolver.resolve_action.assert_called_with(state, action, card_db)

    @patch('dm_ai_module.PhaseManager')
    def test_phase_manager_wrappers(self, mock_phase_manager):
        state = MagicMock()
        card_db = {}

        EngineCompat.PhaseManager_next_phase(state, card_db)
        dm_ai_module.PhaseManager.next_phase.assert_called_with(state, card_db)

if __name__ == '__main__':
    unittest.main()
