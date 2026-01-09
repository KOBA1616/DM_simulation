
import os
import json
import random
import copy
import multiprocessing
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    import dm_ai_module
except ImportError:
    # If module is not found, we might be in a CI/Test env where it's not built yet
    # or path is not set.
    # For now, we assume it's available or we mock it in tests.
    dm_ai_module = None

class DeckIndividual:
    """
    Represents a single deck in the population.
    """
    def __init__(self, deck_id: str, cards: List[int], win_rate: float = 0.0):
        self.deck_id = deck_id
        self.cards = cards
        self.win_rate = win_rate
        self.matchups: Dict[str, float] = {} # opponent_deck_id -> win_rate
        self.games_played = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deck_id": self.deck_id,
            "cards": self.cards,
            "win_rate": self.win_rate,
            "matchups": self.matchups,
            "games_played": self.games_played
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DeckIndividual':
        deck = DeckIndividual(data["deck_id"], data["cards"], data.get("win_rate", 0.0))
        deck.matchups = data.get("matchups", {})
        deck.games_played = data.get("games_played", 0)
        return deck

class PopulationManager:
    """
    Manages the population of decks for the evolution ecosystem.
    Responsible for initialization, saving/loading, and basic population operations.
    """
    def __init__(self, card_db: Dict[int, Any], population_size: int = 20, storage_path: str = "data/population"):
        self.card_db = card_db
        self.population_size = population_size
        self.storage_path = storage_path
        self.population: List[DeckIndividual] = []
        self.generation = 0

        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

    def initialize_random_population(self, candidate_pool_ids: List[int], deck_size: int = 40):
        """
        Initializes the population with random decks created from the candidate pool.
        """
        self.population = []
        self.generation = 0

        for i in range(self.population_size):
            deck_cards = []
            # Simple random selection for now
            # In future, we might want to respect civilization balance or mana curve
            for _ in range(deck_size):
                deck_cards.append(random.choice(candidate_pool_ids))

            deck_id = f"gen0_deck{i:03d}"
            self.population.append(DeckIndividual(deck_id, deck_cards))

    def load_population(self, filename: str = "current_population.json"):
        """
        Loads the population from a JSON file.
        """
        filepath = os.path.join(self.storage_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Population file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.generation = data.get("generation", 0)
        self.population = [DeckIndividual.from_dict(d) for d in data.get("decks", [])]

    def save_population(self, filename: str = "current_population.json"):
        """
        Saves the current population to a JSON file.
        """
        data = {
            "generation": self.generation,
            "timestamp": "", # TODO: Add timestamp
            "decks": [d.to_dict() for d in self.population]
        }

        filepath = os.path.join(self.storage_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_population(self) -> List[DeckIndividual]:
        return self.population

    def update_generation(self, new_population: List[DeckIndividual]):
        """
        Updates the population with a new generation of decks.
        """
        self.population = new_population
        self.generation += 1

    def get_deck_by_id(self, deck_id: str) -> Optional[DeckIndividual]:
        for deck in self.population:
            if deck.deck_id == deck_id:
                return deck
        return None

# --- Day 2 Implementation: Parallel Workers ---

def worker_play_batch(
    bin_path: str,
    card_db_path: str,
    batch_matchups: List[Tuple[str, List[int], str, List[int]]],
    games_per_match: int,
    threads_per_match: int
) -> List[Dict[str, Any]]:
    """
    Worker function to execute a batch of matchups.
    Must be top-level for multiprocessing pickling.

    Args:
        bin_path: Path to directory containing dm_ai_module
        card_db_path: Path to cards.json
        batch_matchups: List of (id1, cards1, id2, cards2)
        games_per_match: Number of games to play per matchup
        threads_per_match: Number of threads to use for ParallelRunner

    Returns:
        List of result dicts: {
            "deck_a_id": str,
            "deck_b_id": str,
            "wins_a": int,
            "wins_b": int,
            "draws": int
        }
    """
    # Setup path for this process
    if bin_path not in sys.path:
        sys.path.append(bin_path)

    try:
        import dm_ai_module
    except ImportError:
        # Should not happen if paths are correct
        return []

    # Load Card DB locally in this process
    try:
        card_db = dm_ai_module.JsonLoader.load_cards(card_db_path)
    except Exception as e:
        print(f"[Worker] Error loading cards: {e}")
        return []

    if not card_db:
        print(f"[Worker] Warning: Loaded 0 cards from {card_db_path}")

    # Initialize Runner
    # Note: ParallelRunner constructor takes (card_db, simulations, threads)
    # But here we are using it for `play_deck_matchup`, so constructor params
    # might effectively be for 'default' behavior or pre-allocation.
    # We'll set sims=100 (irrelevant for play_deck_matchup?) and threads=1 (controlled by arg).
    runner = dm_ai_module.ParallelRunner(card_db, 100, 1)

    results = []

    for id_a, cards_a, id_b, cards_b in batch_matchups:
        # play_deck_matchup returns List[int] of game results (0=None, 1=P1, 2=P2, 3=Draw)
        # Signature: play_deck_matchup(deck_a, deck_b, num_games, threads)
        game_results = runner.play_deck_matchup(cards_a, cards_b, games_per_match, threads_per_match)

        wins_a = game_results.count(1)
        wins_b = game_results.count(2)
        draws = game_results.count(3)

        results.append({
            "deck_a_id": id_a,
            "deck_b_id": id_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws
        })

    return results

class ParallelMatchExecutor:
    """
    Executes matchups in parallel using multiprocessing and C++ ParallelRunner.
    """
    def __init__(self, card_db_path: str, num_workers: int = 4):
        self.card_db_path = card_db_path
        self.num_workers = num_workers

        # Determine bin path for workers
        # Assuming we are in python/training/, bin is in ../../bin relative to this file
        # But this file is imported, so use __file__
        self.bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bin'))

    def execute_matchups(
        self,
        matchups: List[Tuple[str, List[int], str, List[int]]],
        games_per_match: int = 100,
        threads_per_match: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Executes a list of matchups in parallel.
        matchups: List of (deck_id_a, deck_a_cards, deck_id_b, deck_b_cards)
        """
        total_matchups = len(matchups)
        if total_matchups == 0:
            return []

        # Calculate chunk size
        chunk_size = (total_matchups + self.num_workers - 1) // self.num_workers
        chunks = [matchups[i:i + chunk_size] for i in range(0, total_matchups, chunk_size)]

        args_list = [
            (self.bin_path, self.card_db_path, chunk, games_per_match, threads_per_match)
            for chunk in chunks
        ]

        print(f"[ParallelMatchExecutor] Executing {total_matchups} matchups using {self.num_workers} workers...")
        start_time = time.time()

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            # starmap allows passing multiple arguments
            results_nested = pool.starmap(worker_play_batch, args_list)

        # Flatten results
        all_results = [res for sublist in results_nested for res in sublist]

        elapsed = time.time() - start_time
        print(f"[ParallelMatchExecutor] Completed in {elapsed:.2f}s ({len(all_results)} matchups)")

        return all_results
