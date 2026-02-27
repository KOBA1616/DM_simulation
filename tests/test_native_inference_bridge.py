
import unittest
import sys
import os

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.engine.compat import EngineCompat, dm_ai_module, GameState

class TestNativeInferenceBridge(unittest.TestCase):
    def setUp(self):
        # Ensure native is enabled/checked
        EngineCompat._check_module()

    def test_tensor_converter(self):
        """Verify TensorConverter_convert_to_tensor returns expected vector."""
        state = GameState(0)
        # Use JsonLoader to get DB (returns map) instead of CardRegistry
        try:
            card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        except Exception:
            card_db = {} # Fallback

        # Should return a list of floats (zeros in fallback)
        vec = EngineCompat.TensorConverter_convert_to_tensor(state, 0, card_db)

        self.assertIsInstance(vec, list)
        # Check specific length against native constant
        expected_len = getattr(dm_ai_module.TensorConverter, 'INPUT_SIZE', 856)
        self.assertEqual(len(vec), expected_len, f"Tensor vector length should be {expected_len}")
        # Note: Native implementation might not return all zeros if state is populated/initialized default

    def test_register_batch_inference_numpy(self):
        """Verify register_batch_inference_numpy accepts a callback."""
        def dummy_callback(batch):
            return []

        # Should not raise exception
        try:
            EngineCompat.register_batch_inference_numpy(dummy_callback)
        except Exception as e:
            self.fail(f"register_batch_inference_numpy failed with: {e}")

    def test_parallel_runner_lifecycle(self):
        """Verify create_parallel_runner and ParallelRunner_play_games."""
        try:
            card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        except Exception:
            card_db = {}

        sims = 10
        batch_size = 4

        runner = EngineCompat.create_parallel_runner(card_db, sims, batch_size)
        self.assertIsNotNone(runner, "Failed to create ParallelRunner")

        initial_states = [GameState(0) for _ in range(2)]

        def dummy_evaluator(states):
            # Returns list of (policy, value)
            return [([0.0]*600, 0.0) for _ in states]

        # Run play_games
        # Note: In fallback, this might be a simple mock or serial execution
        results = EngineCompat.ParallelRunner_play_games(
            runner,
            initial_states,
            dummy_evaluator,
            temperature=1.0,
            add_noise=False,
            threads=1
        )

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(initial_states))

        # Verify result structure (GameResult-like)
        for res in results:
            self.assertTrue(hasattr(res, 'result'), "Result should have 'result' attribute")
            # Native GameResultInfo uses 'result' field, 'winner' might be legacy or deprecated
            # self.assertTrue(hasattr(res, 'winner'), "Result should have 'winner' attribute")

if __name__ == '__main__':
    unittest.main()
