
import os
import sys
import json
import random
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dm_toolkit.types import CardCounts

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
    def __init__(self, cards_path: str, meta_decks_path: str, output_path: Optional[str] = None) -> None:
        self.cards_path: str = cards_path
        self.meta_decks_path: str = meta_decks_path
        self.output_path: str = output_path or meta_decks_path

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

    def load_meta_decks(self) -> None:
        if os.path.exists(self.meta_decks_path):
            with open(self.meta_decks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.meta_decks = data.get("decks", [])
            print(f"Loaded {len(self.meta_decks)} meta decks.")
        else:
            print("Meta decks file not found. Starting with empty meta.")
            self.meta_decks = []

    def save_meta_decks(self) -> None:
        data = {"decks": self.meta_decks}
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.meta_decks)} decks to {self.output_path}")

    def generate_challenger(self) -> Tuple[List[Any], str]:
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

    def collect_smart_stats(self, deck: List[Any], opponent_deck: List[Any], num_games: int = 5) -> Dict[Any, Dict[str, int]]:
        """
        Runs games using C++ ParallelRunner to collect detailed card statistics.
        Returns aggregated stats dictionary: {card_id: {play: int, resource: int, ...}}
        """
        aggregated = {}

        # Use ParallelRunner's C++ stat collection
        # This function returns std::map<CardID, CardStats> which pybind11 converts to Dict[int, CardStats]
        # We assume CardStats is bound and exposes fields
        raw_stats = self.runner.play_deck_matchup_with_stats(deck, opponent_deck, num_games, 4) # 4 threads

        for cid, stats in raw_stats.items():
            if cid not in aggregated:
                aggregated[cid] = {}

            # Map C++ fields to Python dict format expected by evaluate_deck
            aggregated[cid]['play'] = stats.play_count
            aggregated[cid]['resource'] = stats.mana_usage_count
            # Additional stats if needed
            aggregated[cid]['shield_trigger'] = stats.shield_trigger_count
            aggregated[cid]['played_from_hand'] = stats.played_from_hand_count

        return aggregated

    def evaluate_deck(self, challenger_deck: List[Any], challenger_name: str, num_games: int = 10) -> Tuple[float, float]:
        """Runs the challenger against a sample of the meta."""
        # card_counts will be populated below
        if not self.meta_decks:
            # If no meta, it wins by default, but we should still score it
            # Run self-play or dummy play for stats
            dummy_opp = challenger_deck
            smart_stats = self.collect_smart_stats(challenger_deck, dummy_opp, num_games=4) # Few games for stats

            # Calculate Score
            total_score: float = 0.0
            deck_count = len(challenger_deck)
            # Count copies of each card in deck for normalization
            card_counts: Dict[Any, int] = {}
            for cid in challenger_deck:
                card_counts[cid] = card_counts.get(cid, 0) + 1

            for cid, count in card_counts.items():
                stats = smart_stats.get(cid, {'play': 0, 'resource': 0})
                # Formula: (5 * Play + 2 * Resource) / (Appearance Count)
                # Appearance Count approx = num_games * count
                # NOTE: C++ Aggregated Stats are SUM over all games.
                # So we just divide by total copies in deck * total games?
                # Actually, stats are per-card-ID.
                # If I have 4 copies of Card A, and in 1 game I play 2 of them:
                # C++ tracks total plays of Card A = 2.
                # Total opportunity = 4 copies * num_games.

                opportunity = count * 4 # 4 games
                if opportunity > 0:
                    score = (5 * stats['play'] + 2 * stats['resource']) / opportunity
                    total_score += score

            print(f"Deck '{challenger_name}' Smart Score: {total_score:.2f} (Win Rate: 100% - Default)")
            return 1.0, total_score

        wins = 0
        total_games = 0

        # Sample opponents
        opponents = random.sample(self.meta_decks, min(3, len(self.meta_decks)))

        # We also collect stats against the first opponent for "Smart Scoring"
        stats_opp = opponents[0]["cards"]
        smart_stats_games = 10 # More games for stats = better accuracy
        smart_stats = self.collect_smart_stats(challenger_deck, stats_opp, num_games=smart_stats_games)

        for opp in opponents:
            opp_deck = opp["cards"]

            # Match 1: Challenger as P1
            results1 = self.runner.play_deck_matchup(challenger_deck, opp_deck, num_games // 2, 4)
            wins += results1.count(1)

            # Match 2: Challenger as P2
            results2 = self.runner.play_deck_matchup(opp_deck, challenger_deck, num_games - (num_games // 2), 4)
            wins += results2.count(2)

            total_games += num_games

        win_rate = wins / total_games

        # Calculate Deck Smart Score
        total_smart_score: float = 0.0
        card_counts = {}
        for cid in challenger_deck:
            card_counts[cid] = card_counts.get(cid, 0) + 1

        for cid, count in card_counts.items():
            stats = smart_stats.get(cid, {'play': 0, 'resource': 0})
            opportunity = count * smart_stats_games
            if opportunity > 0:
                # 5 pts for Play, 2 pts for Resource
                score = (5 * stats['play'] + 2 * stats['resource']) / opportunity
                total_smart_score += score

        print(f"Deck '{challenger_name}' Win Rate: {win_rate*100:.1f}% ({wins}/{total_games}), Smart Score: {total_smart_score:.2f}")
        return win_rate, total_smart_score

    def run_generation(self, num_challengers: int = 5, games_per_match: int = 20, min_win_rate: float = 0.55) -> None:
        print(f"--- Starting Generation ---")
        challengers = []
        for _ in range(num_challengers):
            deck, name = self.generate_challenger()
            challengers.append((deck, name))

        accepted_count = 0
        for deck, name in challengers:
            wr, score = self.evaluate_deck(deck, name, games_per_match)

            # Acceptance criteria: Mainly Win Rate, but we could use Score as tie breaker or bonus
            # For now, keep Win Rate as primary gate
            if wr >= min_win_rate:
                print(f"  [ACCEPTED] {name} enters the meta! (Score: {score:.2f})")
                self.meta_decks.append({
                    "name": name,
                    "cards": deck,
                    "score": score
                })
                accepted_count += 1
            else:
                print(f"  [REJECTED] {name} too weak (WR: {wr*100:.1f}%, Score: {score:.2f})")

        # Pruning
        if len(self.meta_decks) > 20:
            print("Pruning meta decks...")
            # Keep top 10 newest
            kept = self.meta_decks[-10:]
            remaining = self.meta_decks[:-10]
            if remaining:
                kept.extend(random.sample(remaining, min(5, len(remaining))))
            self.meta_decks = kept

        if accepted_count > 0:
            self.save_meta_decks()

def main() -> None:
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
