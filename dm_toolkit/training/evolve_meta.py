
import sys
import os
import random
import json
import time
import math
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, TYPE_CHECKING

# Ensure dm_ai_module is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

# --- Mock / Import Handling ---
if TYPE_CHECKING:
    # Provide typing import only for static analysis
    import dm_ai_module  # type: ignore
    HAS_MODULE = True
else:
    try:
        # Prefer the packaged shim if available
        from dm_toolkit import dm_ai_module as _dm_module  # type: ignore
        dm_ai_module = _dm_module
    except Exception:
        try:
            import dm_ai_module as _dm_module  # type: ignore
            dm_ai_module = _dm_module
        except Exception:
            dm_ai_module = None

    if dm_ai_module is None:
        HAS_MODULE = False
        print("Warning: dm_ai_module not found. Running in Mock/Simulation mode.")
    else:
        if not hasattr(dm_ai_module, 'JsonLoader') or not hasattr(dm_ai_module, 'ParallelRunner'):
            HAS_MODULE = False
            print("Warning: dm_ai_module native components (JsonLoader/ParallelRunner) not found.")
        else:
            HAS_MODULE = True


class MockParallelRunner:
    def __init__(self, card_db: Any, num_games: int, num_threads: int) -> None:
        self.card_db = card_db

    def play_deck_matchup(self, deck1: List[int], deck2: List[int], num_games: int, num_threads: int) -> List[int]:
        # Return random results: [p1_wins, p2_wins, draws]
        p1_wins = 0
        p2_wins = 0
        draws = 0
        for _ in range(num_games):
            r = random.random()
            if r < 0.48: p1_wins += 1
            elif r < 0.96: p2_wins += 1
            else: draws += 1
        return [p1_wins, p2_wins, draws]

# --- Data Structures ---

@dataclass
class MetaDeck:
    name: str
    cards: List[int]
    wins: int = 0
    matches: int = 0
    elo: float = 1200.0
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cards": self.cards
        }

# --- Evolution System ---

class DeckEvolutionSystem:
    def __init__(self, card_db_path: str, meta_decks_path: str):
        self.card_db_path = card_db_path
        self.meta_decks_path = meta_decks_path
        self.card_db: dict[int, Any] = {}
        self.valid_card_ids: list[int] = []
        self.population: List[MetaDeck] = []
        self.meta_archetypes: List[Dict] = []

        self.load_resources()

    def load_resources(self) -> None:
        # Load Card DB
        if HAS_MODULE:
            loader = dm_ai_module.JsonLoader()
            # The C++ loader returns a map of objects, we might need to wrap or use it directly
            # For mutation logic, we need python access to IDs.
            # We'll also load via Python JSON for easier attribute access if needed
            self.card_db_obj = loader.load_cards(self.card_db_path)
            self.valid_card_ids = list(self.card_db_obj.keys())
        else:
            with open(self.card_db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle list or dict format
                if isinstance(data, list):
                    for c in data:
                        self.card_db[c['id']] = c
                else:
                    self.card_db = data
            self.valid_card_ids = [int(k) for k in self.card_db.keys()]
            self.card_db_obj = self.card_db # Use dict as obj in mock mode

        # Load Meta Decks
        if os.path.exists(self.meta_decks_path):
            with open(self.meta_decks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.meta_archetypes = data.get("decks", [])

        print(f"Loaded {len(self.valid_card_ids)} cards and {len(self.meta_archetypes)} meta archetypes.")

    def initialize_population(self, pop_size: int = 20) -> None:
        self.population = []

        # 1. Seed with Meta Archetypes
        for arch in self.meta_archetypes:
            self.population.append(MetaDeck(
                name=arch.get("name", "Unknown"),
                cards=arch.get("cards", []),
                elo=1500.0, # Higher starting ELO for known meta
                generation=0
            ))

        # 2. Fill rest with mutations of archetypes or random
        while len(self.population) < pop_size:
            if self.meta_archetypes:
                parent = random.choice(self.meta_archetypes)
                new_deck = MetaDeck(
                    name=f"{parent.get('name')}_Var",
                    cards=list(parent.get("cards", [])),
                    elo=1200.0,
                    generation=0
                )
                self.mutate_deck(new_deck, mutations=4)
                self.population.append(new_deck)
            else:
                # Completely random if no meta
                self.population.append(self.create_random_deck())

        print(f"Population initialized with {len(self.population)} decks.")

    def create_random_deck(self) -> MetaDeck:
        # Simple random deck
        deck: list[int] = []
        while len(deck) < 40:
            cid = random.choice(self.valid_card_ids)
            if deck.count(cid) < 4:
                deck.append(cid)
        return MetaDeck("Random_Starter", deck)

    def mutate_deck(self, meta_deck: MetaDeck, mutations: int = 2) -> None:
        """Randomly swaps N cards in the deck with valid alternatives."""
        deck = meta_deck.cards
        changes = 0
        attempts = 0
        max_attempts = 100

        while changes < mutations and attempts < max_attempts:
            attempts += 1

            # Remove a card
            idx_to_remove = random.randint(0, len(deck) - 1)
            removed_id = deck[idx_to_remove]

            # Pick a new card
            new_id = random.choice(self.valid_card_ids)

            # Constraints
            if new_id == removed_id: continue
            if deck.count(new_id) >= 4: continue

            # Apply
            deck[idx_to_remove] = new_id
            changes += 1

            # Rename if not already variant
            if "Var" not in meta_deck.name:
                meta_deck.name += "_Var"

    def run_evolution(self, generations: int = 5, matches_per_pair: int = 10) -> None:
        # Runner can be either the C++ ParallelRunner or the local MockParallelRunner
        runner: Any
        if HAS_MODULE:
            runner = dm_ai_module.ParallelRunner(self.card_db_obj, 1, 1) # Internal threads managed by play_matchup?
        else:
            runner = MockParallelRunner(self.card_db_obj, 1, 1)

        for gen in range(1, generations + 1):
            print(f"\n=== Generation {gen} ===")

            # 1. Evaluate (Tournament)
            # Simple approach: Each deck plays against N random opponents
            matches_per_deck = 5

            for i, deck_agent in enumerate(self.population):
                opponents = random.sample(self.population, k=matches_per_deck)
                for opp in opponents:
                    if deck_agent == opp: continue

                    # Play Match
                    results = runner.play_deck_matchup(deck_agent.cards, opp.cards, matches_per_pair, 1)
                    p1_wins, p2_wins, draws = results

                    # Update Stats
                    deck_agent.matches += matches_per_pair
                    deck_agent.wins += p1_wins
                    opp.matches += matches_per_pair
                    opp.wins += p2_wins

                    # Update Elo (Simple K-factor)
                    k = 32
                    expected_p1 = 1 / (1 + 10 ** ((opp.elo - deck_agent.elo) / 400))
                    actual_p1 = (p1_wins + 0.5 * draws) / matches_per_pair

                    deck_agent.elo += k * (actual_p1 - expected_p1)
                    opp.elo -= k * (actual_p1 - expected_p1)

            # 2. Sort & Report
            self.population.sort(key=lambda d: d.elo, reverse=True)
            top_deck = self.population[0]
            print(f"Top Deck: {top_deck.name} (Elo: {top_deck.elo:.1f}, WinRate: {top_deck.wins/max(1, top_deck.matches):.2%})")

            # 3. Selection (Keep top 50%)
            cutoff = len(self.population) // 2
            survivors = self.population[:cutoff]

            # 4. Reproduction (Fill bottom 50% with mutated clones of survivors)
            new_population = [copy.deepcopy(s) for s in survivors]

            while len(new_population) < len(self.population):
                parent = random.choice(survivors)
                child = copy.deepcopy(parent)
                child.generation = gen
                child.matches = 0
                child.wins = 0
                # Reset Elo slightly towards mean? Or keep?
                # Keeping Elo helps stability, but reset allows climbing
                child.name = f"{parent.name}_Gen{gen}"
                self.mutate_deck(child, mutations=random.randint(1, 4))
                new_population.append(child)

            self.population = new_population

        # End of Evolution
        self.save_meta_decks()

    def save_meta_decks(self) -> None:
        # Save top 5 unique decks to meta_decks.json
        top_decks = self.population[:5]
        output_data = {"decks": [d.to_dict() for d in top_decks]}

        with open(self.meta_decks_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved top {len(top_decks)} decks to {self.meta_decks_path}")

# --- Entry Point ---

def main() -> None:
    base_dir = os.path.dirname(__file__)
    # Path resolution relative to script location
    # script is in python/training/
    # data is in data/ (which is ../../data relative to script?)
    # Wait, repo root is ../../

    root_dir = os.path.abspath(os.path.join(base_dir, '../../'))
    card_db_path = os.path.join(root_dir, 'data', 'cards.json')
    meta_decks_path = os.path.join(root_dir, 'data', 'meta_decks.json')

    if not os.path.exists(card_db_path):
        # Fallback for running from root
        card_db_path = 'data/cards.json'
        meta_decks_path = 'data/meta_decks.json'

    print(f"Starting Deck Evolution System...")
    print(f"Cards: {card_db_path}")
    print(f"Meta: {meta_decks_path}")

    system = DeckEvolutionSystem(card_db_path, meta_decks_path)

    # Run a short evolution
    system.initialize_population(pop_size=8)
    system.run_evolution(generations=2, matches_per_pair=2)

if __name__ == "__main__":
    main()
