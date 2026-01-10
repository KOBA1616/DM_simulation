
import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add current directory to path to import evolution_ecosystem
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import evolution_ecosystem

class TestEvolutionLoop(unittest.TestCase):
    def setUp(self):
        self.card_db = {1: "Card1", 2: "Card2", 3: "Card3", 4: "Card4"}
        self.storage_path = "test_evolution_data"
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

        self.pop_manager = evolution_ecosystem.PopulationManager(self.card_db, population_size=4, storage_path=self.storage_path)
        self.pop_manager.initialize_random_population(candidate_pool_ids=[1, 2, 3, 4], deck_size=4)

        self.ev_op = evolution_ecosystem.EvolutionOperator(
            candidate_pool_ids=[1, 2, 3, 4],
            mutation_rate=0.5,
            survival_rate=0.5
        )

    def tearDown(self):
        import shutil
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)

    def test_crossover(self):
        p1 = [1, 1, 1, 1]
        p2 = [2, 2, 2, 2]
        child = self.ev_op.crossover(p1, p2)
        self.assertEqual(len(child), 4)
        # With single point crossover, we expect some 1s and some 2s, or all 1s/all 2s if point is 0/4 (unlikely with randint(1, size-1))
        # Logic is randint(1, 3) for size 4.
        self.assertTrue(1 in child or 2 in child)

    def test_mutate(self):
        deck = [1, 1, 1, 1]
        # Mutation rate 0.5, likely to change at least one
        # But we can't guarantee randomness in unit test easily without seeding.
        # Just check length is preserved.
        mutated = self.ev_op.mutate(deck)
        self.assertEqual(len(mutated), 4)
        for c in mutated:
            self.assertIn(c, [1, 2, 3, 4])

    @patch('evolution_ecosystem.ParallelMatchExecutor')
    def test_run_evolution_loop(self, MockExecutor):
        # Setup Mock Executor
        executor_instance = MockExecutor.return_value

        def side_effect_execute(matchups, games, threads):
            results = []
            for m in matchups:
                id_a, _, id_b, _ = m
                # Determine winner deterministically for test stability
                # Let's say deck with lower ID string wins
                if id_a < id_b:
                    wins_a = 10
                    wins_b = 0
                else:
                    wins_a = 0
                    wins_b = 10
                results.append({
                    "deck_a_id": id_a,
                    "deck_b_id": id_b,
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "draws": 0
                })
            return results

        executor_instance.execute_matchups.side_effect = side_effect_execute

        # Run loop
        evolution_ecosystem.run_evolution_loop(
            self.pop_manager,
            executor_instance,
            self.ev_op,
            generations=2,
            games_per_match=10
        )

        # Verify Generation Increased
        self.assertEqual(self.pop_manager.generation, 2)

        # Verify Population Size
        self.assertEqual(len(self.pop_manager.get_population()), 4)

        # Verify files created
        self.assertTrue(os.path.exists(os.path.join(self.storage_path, "gen0_stats.json")))
        self.assertTrue(os.path.exists(os.path.join(self.storage_path, "gen1_stats.json")))
        self.assertTrue(os.path.exists(os.path.join(self.storage_path, "current_population.json")))

    def test_evolve_logic(self):
        # Manually test evolve
        pop = self.pop_manager.get_population()
        # Set win rates: deck 0 is best
        pop[0].win_rate = 1.0
        pop[1].win_rate = 0.5
        pop[2].win_rate = 0.2
        pop[3].win_rate = 0.0

        next_gen = self.ev_op.evolve(pop, 1)

        self.assertEqual(len(next_gen), 4)
        # With survival rate 0.5, top 2 should survive
        survivor_ids = [d.deck_id for d in next_gen[:2]]
        self.assertIn(pop[0].deck_id, survivor_ids)
        self.assertIn(pop[1].deck_id, survivor_ids)

        # Others should be children
        self.assertTrue(next_gen[2].deck_id.startswith("gen1_child"))

if __name__ == '__main__':
    unittest.main()
