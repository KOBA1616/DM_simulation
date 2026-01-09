
import unittest
import os
import json
import shutil
from python.training.evolution_ecosystem import PopulationManager, DeckIndividual

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

if __name__ == '__main__':
    unittest.main()
