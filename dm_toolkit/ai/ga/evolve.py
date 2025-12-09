import os
import sys
import random
import json

# Add root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import dm_ai_module
from dm_toolkit.ai.agent.mcts import MCTS

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
        scores = [0] * self.population_size
        matches_per_deck = 2 # Keep low for dev speed
        
        print("  Evaluating population...")
        for i in range(self.population_size):
            deck_a = self.population[i]
            
            for _ in range(matches_per_deck):
                # Pick opponent
                j = random.randint(0, self.population_size - 1)
                if i == j:
                    j = (j + 1) % self.population_size
                deck_b = self.population[j]
                
                winner_id = self._play_match(deck_a, deck_b, network)
                
                if winner_id == 0:
                    scores[i] += 1
                    
        return scores

    def _play_match(self, deck_a, deck_b, network):
        gs = dm_ai_module.GameState(random.randint(0, 100000))
        gs.set_deck(0, deck_a)
        gs.set_deck(1, deck_b)
        dm_ai_module.PhaseManager.start_game(gs)
        
        mcts = None
        if network:
            mcts = MCTS(network, self.card_db, simulations=10)
            
        turn_count = 0
        while True:
            turn_count += 1
            if turn_count > 200:
                # print("Draw (Turn Limit)")
                return -1 # Draw
                
            is_over, result = dm_ai_module.PhaseManager.check_game_over(gs)
            if is_over:
                # 1=P1(0), 2=P2(1), 3=Draw
                # print(f"Game Over: {result} at turn {turn_count}")
                if result == 1: return 0
                if result == 2: return 1
                return -1
                
            if mcts:
                root = mcts.search(gs)
                # Select best action by visit count
                best_action = None
                max_visits = -1
                for child in root.children:
                    if child.visit_count > max_visits:
                        max_visits = child.visit_count
                        best_action = child.action
                
                if best_action is None:
                    # Fallback if no children (should be game over or pass)
                    actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, self.card_db)
                    if not actions:
                        dm_ai_module.PhaseManager.next_phase(gs)
                        continue
                    action = actions[0]
                else:
                    action = best_action
            else:
                # Random
                actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, self.card_db)
                if not actions:
                    dm_ai_module.PhaseManager.next_phase(gs)
                    continue
                action = random.choice(actions)
                
            dm_ai_module.EffectResolver.resolve_action(gs, action, self.card_db)
            if action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(gs)
        
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
            # For now, passing None for network (Random Play)
            # In real training, we would pass the trained AlphaZeroNetwork
            scores = self.evaluate(None)
            print(f"  Scores: {scores}")
            
            # Select & Mutate
            # Sort by score
            pop_scores = list(zip(self.population, scores))
            pop_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 50%
            half = max(1, self.population_size // 2)
            survivors = [x[0] for x in pop_scores[:half]]
            
            new_pop = []
            # Elitism: Keep best
            new_pop.extend([d.copy() for d in survivors])
            
            # Fill rest with mutated versions of survivors
            while len(new_pop) < self.population_size:
                parent = random.choice(survivors)
                child = self.mutate(parent)
                new_pop.append(child)
                
            self.population = new_pop

if __name__ == "__main__":
    # Test
    card_db = dm_ai_module.CsvLoader.load_cards("data/cards.csv")
    evo = DeckEvolution(card_db)
    evo.evolve(generations=3)
