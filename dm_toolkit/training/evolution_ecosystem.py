
import os
import sys
import json
import random
import time
import argparse
from datetime import datetime

# Set up paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Ensure it is built and in bin/")
    sys.exit(1)

class EvolutionEcosystem:
    def __init__(self, cards_path, meta_decks_path, output_path=None):
        self.cards_path = cards_path
        self.meta_decks_path = meta_decks_path
        self.output_path = output_path or meta_decks_path

        # Load Data
        print(f"Loading cards from {cards_path}...")
        self.card_db = dm_ai_module.JsonLoader.load_cards(cards_path)
        print(f"Loaded {len(self.card_db)} cards.")

        self.load_meta_decks()

        # Evolution Config
        self.evo_config = dm_ai_module.DeckEvolutionConfig()
        self.evo_config.target_deck_size = 40
        self.evo_config.mutation_rate = 0.2  # 20% mutation chance per card slot

        self.evolver = dm_ai_module.DeckEvolution(self.card_db)

        # Runner
        # We use heuristic agent for deck evaluation for speed
        self.runner = dm_ai_module.ParallelRunner(self.card_db, 50, 1)

    def load_meta_decks(self):
        if os.path.exists(self.meta_decks_path):
            with open(self.meta_decks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.meta_decks = data.get("decks", [])
            print(f"Loaded {len(self.meta_decks)} meta decks.")
        else:
            print("Meta decks file not found. Starting with empty meta.")
            self.meta_decks = []

    def save_meta_decks(self):
        data = {"decks": self.meta_decks}
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.meta_decks)} decks to {self.output_path}")

    def generate_challenger(self):
        """Generates a challenger deck by mutating an existing meta deck or random creation."""
        if not self.meta_decks:
            # Create random deck if no meta
            # Filter for valid cards (creatures/spells) - simplistic approach
            valid_ids = list(self.card_db.keys())
            # Basic deck: 40 random cards
            deck = [random.choice(valid_ids) for _ in range(40)]
            return deck, "Random_Gen0"

        # Select parent
        parent_entry = random.choice(self.meta_decks)
        parent_deck = parent_entry["cards"]
        parent_name = parent_entry["name"]

        # Create pool (all cards)
        pool = list(self.card_db.keys())

        # Evolve
        new_deck = self.evolver.evolve_deck(parent_deck, pool, self.evo_config)

        # Name
        new_name = f"{parent_name}_v{int(time.time()) % 10000}"
        return new_deck, new_name

    def evaluate_deck(self, challenger_deck, challenger_name, num_games=10):
        """Runs the challenger against a sample of the meta."""
        if not self.meta_decks:
            return 1.0 # If no meta, it wins by default (first settler)

        wins = 0
        total_games = 0

        # Sample opponents (e.g. 3 random meta decks)
        opponents = random.sample(self.meta_decks, min(3, len(self.meta_decks)))

        for opp in opponents:
            opp_deck = opp["cards"]
            # Play Match (half as P1, half as P2)
            # ParallelRunner.play_deck_matchup returns [1, 2, 0, 1, ...] (1=P1 win, 2=P2 win)

            # Match 1: Challenger as P1
            results1 = self.runner.play_deck_matchup(challenger_deck, opp_deck, num_games // 2, 4)
            wins += results1.count(1)

            # Match 2: Challenger as P2
            results2 = self.runner.play_deck_matchup(opp_deck, challenger_deck, num_games - (num_games // 2), 4)
            wins += results2.count(2)

            total_games += num_games

        win_rate = wins / total_games
        print(f"Deck '{challenger_name}' Win Rate: {win_rate*100:.1f}% ({wins}/{total_games})")
        return win_rate

    def run_generation(self, num_challengers=5, games_per_match=20, min_win_rate=0.55):
        print(f"--- Starting Generation ---")
        challengers = []
        for _ in range(num_challengers):
            deck, name = self.generate_challenger()
            challengers.append((deck, name))

        accepted_count = 0
        for deck, name in challengers:
            wr = self.evaluate_deck(deck, name, games_per_match)
            if wr >= min_win_rate:
                print(f"  [ACCEPTED] {name} enters the meta!")
                self.meta_decks.append({
                    "name": name,
                    "cards": deck
                })
                accepted_count += 1
            else:
                print(f"  [REJECTED] {name} too weak.")

        # Pruning: If meta is too big (>10), remove lowest win-rate ones?
        # For now, simplistic pruning: remove random old ones if > 20
        if len(self.meta_decks) > 20:
            print("Pruning meta decks...")
            # Ideally we re-evaluate all, but for speed we just keep the newest ones + random
            # Keep top 10 newest
            kept = self.meta_decks[-10:]
            remaining = self.meta_decks[:-10]
            if remaining:
                kept.extend(random.sample(remaining, min(5, len(remaining))))
            self.meta_decks = kept

        if accepted_count > 0:
            self.save_meta_decks()

def main():
    parser = argparse.ArgumentParser(description="Self-Evolution Ecosystem")
    parser.add_argument("--cards", default="data/cards.json", help="Path to cards.json")
    parser.add_argument("--meta", default="data/meta_decks.json", help="Path to meta_decks.json")
    parser.add_argument("--generations", type=int, default=1, help="Number of generations to run")
    parser.add_argument("--challengers", type=int, default=5, help="Challengers per generation")

    args = parser.parse_args()

    if not os.path.exists(args.cards):
        print(f"Cards file not found: {args.cards}")
        return

    ecosystem = EvolutionEcosystem(args.cards, args.meta)

    for i in range(args.generations):
        print(f"\n=== Generation {i+1}/{args.generations} ===")
        ecosystem.run_generation(num_challengers=args.challengers)

if __name__ == "__main__":
    main()
