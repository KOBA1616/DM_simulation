
import unittest
import os
import json
import shutil
from collections import Counter
from python.training.evolution_ecosystem import PopulationManager, DeckIndividual, EvolutionOperator

class TestPopulationManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "temp_test_population"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # Mock Card DB (just ids are needed for now)
        self.mock_card_db = {
            1: "Card A",
            2: "Card B",
            3: "Card C"
        }

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialize_random_population(self):
        manager = PopulationManager(self.mock_card_db, population_size=5, storage_path=self.test_dir)
        candidate_pool = [1, 2, 3]

        manager.initialize_random_population(candidate_pool, deck_size=10)

        self.assertEqual(len(manager.population), 5)
        self.assertEqual(manager.generation, 0)

        for deck in manager.population:
            self.assertEqual(len(deck.cards), 10)
            self.assertTrue(all(c in candidate_pool for c in deck.cards))
            self.assertTrue(deck.deck_id.startswith("gen0_deck"))

            # Check 4-copy rule (though with pool of 3 cards and size 10, it will hit limits)
            # Actually, the implementation loops until deck is full.
            # If pool is small, it might hang or fill with what's available?
            # The implementation:
            # while len(deck_cards) < deck_size:
            #    if current_counts[card_id] < 4: ...
            # With only 3 cards available, max deck size is 12 (4*3).
            # deck_size=10 is fine.
            counts = Counter(deck.cards)
            for c in counts:
                self.assertLessEqual(counts[c], 4)

    def test_save_and_load_population(self):
        manager = PopulationManager(self.mock_card_db, population_size=2, storage_path=self.test_dir)

        # Manually create a population
        deck1 = DeckIndividual("deck1", [1, 1, 2], win_rate=0.5)
        deck2 = DeckIndividual("deck2", [3, 3, 2], win_rate=0.8)
        manager.population = [deck1, deck2]
        manager.generation = 10

        # Save
        filename = "test_pop.json"
        manager.save_population(filename)

        # Check file existence
        filepath = os.path.join(self.test_dir, filename)
        self.assertTrue(os.path.exists(filepath))

        # Create new manager and load
        manager2 = PopulationManager(self.mock_card_db, population_size=2, storage_path=self.test_dir)
        manager2.load_population(filename)

        self.assertEqual(manager2.generation, 10)
        self.assertEqual(len(manager2.population), 2)

        loaded_deck1 = manager2.get_deck_by_id("deck1")
        self.assertIsNotNone(loaded_deck1)
        self.assertEqual(loaded_deck1.cards, [1, 1, 2])
        self.assertEqual(loaded_deck1.win_rate, 0.5)

    def test_update_generation(self):
        manager = PopulationManager(self.mock_card_db, population_size=2, storage_path=self.test_dir)
        manager.generation = 1

        new_decks = [
            DeckIndividual("gen2_1", [1]),
            DeckIndividual("gen2_2", [2])
        ]

        manager.update_generation(new_decks)

        self.assertEqual(manager.generation, 2)
        self.assertEqual(len(manager.population), 2)
        self.assertEqual(manager.population[0].deck_id, "gen2_1")

class TestEvolutionOperator(unittest.TestCase):
    def setUp(self):
        self.candidate_pool = [1, 2, 3, 4, 5]
        self.op = EvolutionOperator(self.candidate_pool)

    def test_select_survivors(self):
        pop = [
            DeckIndividual("d1", [], 0.1),
            DeckIndividual("d2", [], 0.9),
            DeckIndividual("d3", [], 0.5),
            DeckIndividual("d4", [], 0.3)
        ]
        survivors = self.op.select_survivors(pop, selection_rate=0.5)
        self.assertEqual(len(survivors), 2)
        self.assertEqual(survivors[0].deck_id, "d2") # 0.9
        self.assertEqual(survivors[1].deck_id, "d3") # 0.5

    def test_crossover(self):
        # Parents with 10 cards
        p1 = [1]*4 + [2]*4 + [3]*2 # 4x1, 4x2, 2x3
        p2 = [3]*4 + [4]*4 + [5]*2 # 4x3, 4x4, 2x5

        child = self.op.crossover(p1, p2)
        self.assertEqual(len(child), 10)

        # Check 4-copy rule
        counts = Counter(child)
        for c in counts:
            self.assertLessEqual(counts[c], 4)

    def test_mutate(self):
        deck = [1]*4 + [2]*4 + [3]*2
        # Force mutation
        self.op.mutation_rate = 1.0
        mutated = self.op.mutate(deck)

        self.assertEqual(len(mutated), 10)

        # Check that it's different (high probability)
        # Check 4-copy rule
        counts = Counter(mutated)
        for c in counts:
            self.assertLessEqual(counts[c], 4)

    def test_create_next_generation(self):
        pop = [
            DeckIndividual("d1", [1]*10, 0.9), # Top
            DeckIndividual("d2", [2]*10, 0.8), # Top
            DeckIndividual("d3", [3]*10, 0.1),
            DeckIndividual("d4", [4]*10, 0.0)
        ]
        # selection 50% -> d1, d2 survive

        next_gen = self.op.create_next_generation(pop, generation_idx=2, target_size=4)

        self.assertEqual(len(next_gen), 4)

        # First 2 should be survivors (clones of d1, d2) with reset stats
        self.assertEqual(next_gen[0].deck_id, "d1")
        self.assertEqual(next_gen[0].win_rate, 0.0)
        self.assertEqual(next_gen[1].deck_id, "d2")

        # Next 2 should be children (new IDs)
        self.assertTrue(next_gen[2].deck_id.startswith("gen2_deck"))
        self.assertTrue(next_gen[3].deck_id.startswith("gen2_deck"))

if __name__ == '__main__':
    unittest.main()
