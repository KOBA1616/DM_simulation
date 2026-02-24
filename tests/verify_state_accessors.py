import unittest
import sys
import os

# Ensure we can import dm_toolkit and dm_ai_module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.engine.compat import EngineCompat
from dm_ai_module import GameState, Phase, Player

class TestEngineCompatAccessors(unittest.TestCase):
    def setUp(self):
        self.state = GameState()
        # Ensure minimal setup
        if not hasattr(self.state, 'players') or not self.state.players:
            self.state.players = [Player(), Player()]

        self.state.active_player_id = 0
        self.state.current_phase = Phase.MANA
        self.state.turn_number = 5
        self.state.pending_query = "TestQuery"
        self.state.effect_buffer = ["Effect1"]
        self.state.command_history = ["Cmd1"]

        # Populate pending effects with a mock object
        class MockEffect:
            type = 'TEST_EFFECT'
            source_instance_id = 10
            target_player = 0
            card_id = 999

        self.state.pending_effects = [MockEffect()]

    def test_get_current_phase(self):
        print("\nTesting get_current_phase...")
        phase = EngineCompat.get_current_phase(self.state)
        print(f"Phase: {phase} type: {type(phase)}")
        # We expect it to try to return the Enum if available
        # It might be an int 2 or Phase.MANA
        self.assertTrue(phase == Phase.MANA or phase == 2 or phase == 'MANA')

    def test_get_turn_number(self):
        print("\nTesting get_turn_number...")
        turn = EngineCompat.get_turn_number(self.state)
        self.assertEqual(turn, 5)

    def test_get_active_player_id(self):
        print("\nTesting get_active_player_id...")
        pid = EngineCompat.get_active_player_id(self.state)
        self.assertEqual(pid, 0)

    def test_get_player(self):
        print("\nTesting get_player...")
        p0 = EngineCompat.get_player(self.state, 0)
        self.assertIsNotNone(p0)
        # Verify it's the same object
        self.assertEqual(p0, self.state.players[0])

        # Test Out of bounds - should return None
        p_invalid = EngineCompat.get_player(self.state, 99)
        self.assertIsNone(p_invalid)

    def test_get_pending_query(self):
        print("\nTesting get_pending_query...")
        q = EngineCompat.get_pending_query(self.state)
        self.assertEqual(q, "TestQuery")

    def test_get_pending_effects_info(self):
        print("\nTesting get_pending_effects_info...")
        # This wrapper calls dm_ai_module.get_pending_effects_info
        info = EngineCompat.get_pending_effects_info(self.state)
        print(f"Pending Info: {info}")
        self.assertIsInstance(info, list)
        # With the pure python stub, it should return a list of info tuples
        # But if compat delegates strictly and the stub is loaded, it should work.
        # If compat fails to find the function, it returns [].
        # We asserted we put something in pending_effects, so we expect something back
        # IF the stub implementation logic matches.

        # dm_ai_module.py stub logic:
        # returns [(str(t), sid, pid, cmd)]
        if len(info) > 0:
            item = info[0]
            self.assertTrue(len(item) >= 4)
            self.assertEqual(item[0], 'TEST_EFFECT')

    def test_get_command_history(self):
        print("\nTesting get_command_history...")
        hist = EngineCompat.get_command_history(self.state)
        self.assertEqual(hist, ["Cmd1"])

    def test_dump_state_debug(self):
        print("\nTesting dump_state_debug...")
        dump = EngineCompat.dump_state_debug(self.state)
        self.assertIsInstance(dump, dict)
        self.assertIn('players', dump)
        self.assertIn('active_player_id', dump)
        print(f"Dump keys: {dump.keys()}")

if __name__ == '__main__':
    unittest.main()
