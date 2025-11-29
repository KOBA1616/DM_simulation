import os
import sys
import random
import json
import dm_ai_module

class DeckEvolution:
    def __init__(self, card_db):
        self.card_db = card_db
        self.population_size = 10
        self.population = [] # List of deck lists (list of card IDs)
        self.generation_dir = "data/generations"
        os.makedirs(self.generation_dir, exist_ok=True)
        
    def initialize_population(self):
        # Create random decks
        all_ids = list(self.card_db.keys())
        for _ in range(self.population_size):
            deck = []
            for _ in range(40):
                deck.append(random.choice(all_ids))
            self.population.append(deck)
            
    def evaluate(self, network):
        # Run league
        scores = [0] * self.population_size
        # TODO: Implement league play using self_play logic with fixed decks
        return scores
        
    def mutate(self, deck):
        # Swap 1-4 cards
        new_deck = deck.copy()
        num_swaps = random.randint(1, 4)
        all_ids = list(self.card_db.keys())
        
        for _ in range(num_swaps):
            idx = random.randint(0, 39)
            new_card = random.choice(all_ids)
            new_deck[idx] = new_card
            
        return new_deck
        
    def save_generation(self, gen_idx):
        data = {
            "generation": gen_idx,
            "decks": self.population
        }
        path = os.path.join(self.generation_dir, f"gen_{gen_idx:03d}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Saved generation {gen_idx} to {path}")
        
    def evolve(self, generations=10):
        self.initialize_population()
        
        for gen in range(generations):
            print(f"Generation {gen}")
            self.save_generation(gen)
            
            # Evaluate
            # scores = self.evaluate(None)
            
            # Select & Mutate (Simple elitism + mutation for now)
            # Just mutate everyone for testing flow
            new_pop = []
            for deck in self.population:
                new_pop.append(self.mutate(deck))
            self.population = new_pop

if __name__ == "__main__":
    # Test
    card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
    evo = DeckEvolution(card_db)
    evo.evolve(generations=3)
