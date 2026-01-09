
import os
import json
import random
import copy
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    import dm_ai_module
except ImportError:
    # If module is not found, we might be in a CI/Test env where it's not built yet
    # or path is not set.
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

class ParallelWorkers:
    """
    Handles parallel execution of matches between decks.
    """
    def __init__(self, card_db: Dict[int, Any], num_workers: int = 4, sims: int = 100):
        self.card_db = card_db
        self.num_workers = num_workers
        self.sims = sims # MCTS simulations per move
        self.batch_size = 32

    def run_matchups(self, matchups: List[Tuple[DeckIndividual, DeckIndividual]], games_per_matchup: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Runs matchups in parallel.
        Returns a dictionary mapping deck_id -> opponent_id -> win_rate (from deck_id perspective).
        """
        results: Dict[str, Dict[str, float]] = {}

        if dm_ai_module is None:
            print("Warning: dm_ai_module not available. Skipping matches.")
            return {}

        runner = dm_ai_module.ParallelRunner(self.card_db, self.sims, self.batch_size)

        # Pre-initialize results structure
        for d1, d2 in matchups:
            if d1.deck_id not in results: results[d1.deck_id] = {}
            if d2.deck_id not in results: results[d2.deck_id] = {}

        total_matchups = len(matchups)
        print(f"Running {total_matchups} matchups with {games_per_matchup} games each...")

        for idx, (deck1, deck2) in enumerate(matchups):
            # deck1 vs deck2
            # play_deck_matchup returns list of game results (1=P1_WIN, 2=P2_WIN, 3=DRAW)
            # The signature is play_deck_matchup(deck1_cards, deck2_cards, num_games, num_threads)

            # Using OpenMP threads inside ParallelRunner
            # We use self.num_workers as the number of threads for this matchup
            match_results = runner.play_deck_matchup(deck1.cards, deck2.cards, games_per_matchup, self.num_workers)

            p1_wins = 0
            p2_wins = 0
            draws = 0

            for res in match_results:
                if res == 1: p1_wins += 1
                elif res == 2: p2_wins += 1
                else: draws += 1

            # Calculate win rates (ignoring draws for simplicity or counting as 0.5?)
            # Usually win rate = wins / total
            total_games = len(match_results)
            if total_games > 0:
                wr_p1 = p1_wins / total_games
                wr_p2 = p2_wins / total_games
            else:
                wr_p1 = 0.0
                wr_p2 = 0.0

            results[deck1.deck_id][deck2.deck_id] = wr_p1
            results[deck2.deck_id][deck1.deck_id] = wr_p2

            if (idx + 1) % 5 == 0:
                print(f"Completed {idx + 1}/{total_matchups} matchups.")

        return results

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
        self.workers = ParallelWorkers(card_db)

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
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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

    def evaluate_population(self, games_per_matchup: int = 10):
        """
        Evaluates the current population by running round-robin (or random sample) matchups.
        Updates win_rate and matchups for each individual.
        """
        matchups = []
        # Round Robin for small population
        # For N=20, matches = 20*19/2 = 190. Feasible.

        decks = self.population
        for i in range(len(decks)):
            for j in range(i + 1, len(decks)):
                matchups.append((decks[i], decks[j]))

        results = self.workers.run_matchups(matchups, games_per_matchup)

        # Update individuals
        for deck in self.population:
            if deck.deck_id in results:
                opp_results = results[deck.deck_id]
                deck.matchups.update(opp_results)

                # Calculate average win rate
                if len(deck.matchups) > 0:
                    deck.win_rate = sum(deck.matchups.values()) / len(deck.matchups)
                else:
                    deck.win_rate = 0.0

                # Assuming games_played tracks total games
                # Each matchup entry represents `games_per_matchup` games roughly
                deck.games_played = len(deck.matchups) * games_per_matchup

    def evolve_step(self):
        """
        Executes one step of evolution (Evaluation -> Selection -> Mutation -> Next Gen).
        This is a placeholder for Day 3.
        """
        print(f"Starting Evolution Step for Generation {self.generation}")
        self.evaluate_population()
        # TODO: Implement Selection and Mutation (Day 3)
        self.save_population(f"gen{self.generation:03d}.json")
