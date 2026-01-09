import sys
import os
import unittest
import json
import random
from unittest.mock import MagicMock, patch

# Adjust path to find modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python'))

# Mock dm_ai_module globally BEFORE importing the module under test
# This ensures that even if the real module exists, we use the mock
mock_dm = MagicMock()
sys.modules['dm_ai_module'] = mock_dm

# Configure the mock
mock_runner = MagicMock()
mock_dm.ParallelRunner.return_value = mock_runner
# play_deck_matchup returns list of ints (results)
# 1=P1_WIN, 2=P2_WIN, 3=DRAW
mock_runner.play_deck_matchup.return_value = [1, 2, 1, 3, 1]

from training.evolution_ecosystem import PopulationManager, DeckIndividual, ParallelWorkers

class TestEvolutionEcosystem(unittest.TestCase):
    def setUp(self):
        self.card_db = {1: "CardA", 2: "CardB"} # Dummy DB
        self.manager = PopulationManager(self.card_db, population_size=4, storage_path="test_data/population")

        # Candidate pool
        self.pool = [1, 2]

    def test_initialization(self):
        self.manager.initialize_random_population(self.pool, deck_size=10)
        self.assertEqual(len(self.manager.population), 4)
        self.assertEqual(len(self.manager.population[0].cards), 10)
        self.assertEqual(self.manager.generation, 0)

    def test_parallel_workers_run_matchups(self):
        workers = ParallelWorkers(self.card_db, num_workers=2, sims=10)

        d1 = DeckIndividual("d1", [1]*10)
        d2 = DeckIndividual("d2", [2]*10)

        matchups = [(d1, d2)]

        # With mocked dm_ai_module, this should return our mocked results
        results = workers.run_matchups(matchups, games_per_matchup=5)

        # d1 vs d2 -> [1, 2, 1, 3, 1] -> P1 wins 3/5 = 0.6, P2 wins 1/5 = 0.2
        # d1 win rate vs d2 should be 0.6
        self.assertIn("d1", results)
        self.assertIn("d2", results["d1"])
        self.assertAlmostEqual(results["d1"]["d2"], 0.6)

        # d2 win rate vs d1 should be 0.2
        self.assertIn("d1", results["d2"])
        self.assertAlmostEqual(results["d2"]["d1"], 0.2)

    def test_evaluate_population(self):
        self.manager.initialize_random_population(self.pool, deck_size=10)

        self.manager.evaluate_population(games_per_matchup=5)

        for deck in self.manager.population:
            # Each deck plays against 3 others
            self.assertEqual(len(deck.matchups), 3)
            self.assertTrue(0.0 <= deck.win_rate <= 1.0)
            self.assertEqual(deck.games_played, 15)

    def tearDown(self):
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")

if __name__ == '__main__':
    unittest.main()
