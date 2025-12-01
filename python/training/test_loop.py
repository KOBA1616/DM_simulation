
import os
import sys
import unittest

# Ensure bin is in path (absolute path)
bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bin'))
if bin_path not in sys.path:
    sys.path.append(bin_path)

import dm_ai_module

class TestLoopDetection(unittest.TestCase):
    def setUp(self):
        # Dummy DB
        self.card_db = {}
        # Create a simple game state
        self.instance = dm_ai_module.GameInstance(42, self.card_db)
        self.state = self.instance.state

    def test_loop_detection(self):
        # Verify initial winner
        self.assertEqual(self.state.winner, dm_ai_module.GameResult.NONE)

        # Manually trigger loop detection by mocking history using DevTools
        # We need to simulate the same state occurring 3 times.
        # DevTools.trigger_loop_detection(state) pushes current hash 2 times to history.
        # Then the next update_loop_check (called by check_game_over) should find 2 history + 1 current = 3.

        dm_ai_module.DevTools.trigger_loop_detection(self.state)

        # Now trigger check_game_over
        is_over, result = dm_ai_module.PhaseManager.check_game_over(self.state)

        # Should be over and winner should be set
        self.assertTrue(is_over)
        # Active player is 0, so P1_WIN (enum value 1)
        # The enum value is returned as int by check_game_over wrapper if not cast, but pybind11 usually handles enums.
        # The error `1 != <GameResult.P1_WIN: 1>` implies result is int 1, but we compare with Enum.
        # PhaseManager.check_game_over wrapper returns (bool, int) in bindings.cpp
        # Let's check bindings.cpp

        self.assertEqual(result, int(dm_ai_module.GameResult.P1_WIN))
        self.assertEqual(self.state.winner, dm_ai_module.GameResult.P1_WIN)

if __name__ == '__main__':
    unittest.main()
