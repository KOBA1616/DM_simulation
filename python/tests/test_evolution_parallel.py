
import unittest
import os
import sys
import shutil
import json

# Add bin and python to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.evolution_ecosystem import ParallelMatchExecutor, DeckIndividual, PopulationManager

class TestParallelExecutor(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/test_evolution"
        os.makedirs(self.test_dir, exist_ok=True)
        self.cards_path = os.path.join(self.test_dir, "cards.json")

        # Create dummy cards.json if not exists (or rely on system one if reliable)
        # Using system one "data/cards.json" might be safer if dummy fails loading
        # But we saw data/cards.json loading 0 cards.
        # Let's try to use "data/cards.json" assuming the environment is fixed or we mock.

        if os.path.exists("data/cards.json"):
            self.cards_path = "data/cards.json"
        else:
            # Create a minimal valid cards.json
            pass

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_executor_instantiation(self):
        executor = ParallelMatchExecutor(self.cards_path, num_workers=2)
        self.assertEqual(executor.num_workers, 2)
        self.assertTrue(os.path.exists(executor.bin_path))

    def test_execution_flow(self):
        """
        Tests the execution flow.
        Note: If dm_ai_module is broken (JsonLoader returns 0 cards),
        the worker will likely return empty results or 0 wins.
        This test mainly checks the multiprocessing wiring.
        """
        executor = ParallelMatchExecutor(self.cards_path, num_workers=2)

        # Define dummy matchups
        # Using card ID 1 (New Card) if it exists
        deck_a = [1] * 40
        deck_b = [1] * 40

        matchups = [
            ("deck1", deck_a, "deck2", deck_b),
            ("deck2", deck_b, "deck1", deck_a)
        ]

        # Run small batch
        results = executor.execute_matchups(matchups, games_per_match=2, threads_per_match=1)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["deck_a_id"], "deck1")
        self.assertEqual(results[0]["deck_b_id"], "deck2")
        self.assertTrue("wins_a" in results[0])

if __name__ == '__main__':
    unittest.main()
