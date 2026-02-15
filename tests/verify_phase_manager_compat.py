import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dm_ai_module
from dm_toolkit.engine.compat import EngineCompat

class TestPhaseManagerCompat(unittest.TestCase):
    def setUp(self):
        self.state = dm_ai_module.GameState()
        self.card_db = {}
        # Ensure clean state
        self.state.current_phase = 0
        self.state.game_over = False
        self.state.winner = dm_ai_module.GameResult.NONE

    def test_standard_lifecycle(self):
        """Test the standard start_game and phase cycling behavior (Happy Path)."""
        print("\n[Test] Standard Lifecycle")

        # 1. Start Game
        EngineCompat.PhaseManager_start_game(self.state, self.card_db)

        # Expectation: Start game sets phase to MANA (2)
        # Note: dm_ai_module.Phase.MANA is 2
        print(f"  Post-Start Phase: {self.state.current_phase}")
        self.assertEqual(self.state.current_phase, dm_ai_module.Phase.MANA)

        # 2. Cycle Phases
        # MANA(2) -> MAIN(3)
        EngineCompat.PhaseManager_next_phase(self.state, self.card_db)
        print(f"  Phase after 1st next: {self.state.current_phase}")
        self.assertEqual(self.state.current_phase, dm_ai_module.Phase.MAIN)

        # MAIN(3) -> ATTACK(4)
        EngineCompat.PhaseManager_next_phase(self.state, self.card_db)
        print(f"  Phase after 2nd next: {self.state.current_phase}")
        self.assertEqual(self.state.current_phase, dm_ai_module.Phase.ATTACK)

        # ATTACK(4) -> END(5)
        EngineCompat.PhaseManager_next_phase(self.state, self.card_db)
        print(f"  Phase after 3rd next: {self.state.current_phase}")
        self.assertEqual(self.state.current_phase, dm_ai_module.Phase.END)

        # END(5) -> MANA(2) (Next Turn)
        EngineCompat.PhaseManager_next_phase(self.state, self.card_db)
        print(f"  Phase after 4th next: {self.state.current_phase}")
        self.assertEqual(self.state.current_phase, dm_ai_module.Phase.MANA)

    def test_forced_progression(self):
        """Test that EngineCompat forces progression if the native engine is stuck."""
        print("\n[Test] Forced Progression")

        # Set initial phase
        self.state.current_phase = dm_ai_module.Phase.MANA

        # Monkeypatch dm_ai_module.PhaseManager.next_phase to do nothing
        original_next_phase = dm_ai_module.PhaseManager.next_phase
        dm_ai_module.PhaseManager.next_phase = MagicMock(return_value=None)

        try:
            # Call compat wrapper
            # It should detect no change, retry 3 times, then force it.
            EngineCompat.PhaseManager_next_phase(self.state, self.card_db)

            print(f"  Phase after stuck next: {self.state.current_phase}")

            # Verify the native mock was called (retries)
            # It is called 1 (initial) + 3 (retries) = 4 times roughly
            self.assertGreaterEqual(dm_ai_module.PhaseManager.next_phase.call_count, 1)

            # Verify the phase actually changed despite native doing nothing
            self.assertEqual(self.state.current_phase, dm_ai_module.Phase.MAIN)

        finally:
            # Restore
            dm_ai_module.PhaseManager.next_phase = original_next_phase

    def test_hang_detection(self):
        """Test that EngineCompat raises RuntimeError if phase refuses to change even after force."""
        print("\n[Test] Hang Detection")

        # Set initial phase
        self.state.current_phase = dm_ai_module.Phase.MANA

        # Monkeypatch dm_ai_module.PhaseManager.next_phase to do nothing
        original_next_phase = dm_ai_module.PhaseManager.next_phase
        dm_ai_module.PhaseManager.next_phase = MagicMock(return_value=None)

        # Create a "Stubborn" state object that ignores setattr for current_phase
        class StubbornState:
            def __init__(self):
                self.current_phase = dm_ai_module.Phase.MANA
                self.active_player_id = 0
                self._phase_nochange_count = 0

            def __setattr__(self, key, value):
                if key == 'current_phase':
                    # Ignore changes to phase
                    pass
                else:
                    super().__setattr__(key, value)

        stubborn_state = StubbornState()

        try:
            # We expect a RuntimeError after ~15 attempts
            # Since the internal counter persists on the state object,
            # we need to loop enough times.
            # The compat function increments the counter every time it fails to change phase.

            with self.assertRaises(RuntimeError) as cm:
                for i in range(20):
                    EngineCompat.PhaseManager_next_phase(stubborn_state, self.card_db)

            print(f"  Caught expected error: {cm.exception}")
            self.assertIn("phase did not advance", str(cm.exception))

        finally:
            dm_ai_module.PhaseManager.next_phase = original_next_phase

    def test_game_over_check(self):
        """Test PhaseManager_check_game_over normalization."""
        print("\n[Test] Game Over Check")

        # Case 1: Not Over
        self.state.game_over = False
        is_over, res = EngineCompat.PhaseManager_check_game_over(self.state)
        self.assertFalse(is_over)

        # Case 2: Over
        self.state.game_over = True
        self.state.winner = dm_ai_module.GameResult.P1_WIN

        is_over, res = EngineCompat.PhaseManager_check_game_over(self.state)
        self.assertTrue(is_over)

        # Check result object
        if res is not None:
            # If it returns a GameResult object (which acts like int in stub)
            print(f"  Game Result: {res}")
            # In stub, GameResult is an IntEnum subclass or int
            try:
                val = int(res)
                self.assertEqual(val, dm_ai_module.GameResult.P1_WIN)
            except:
                # Might be an object with .result
                self.assertEqual(res.result, dm_ai_module.GameResult.P1_WIN)

if __name__ == '__main__':
    unittest.main()
