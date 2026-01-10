
import os
import json
import random
import copy
import multiprocessing
import sys
import time
import itertools
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

    runner = dm_ai_module.ParallelRunner(card_db, 100, 1)

    results = []

    for id_a, cards_a, id_b, cards_b in batch_matchups:
        # play_deck_matchup returns List[int] of game results (0=None, 1=P1, 2=P2, 3=Draw)
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

# --- Day 3 Implementation: Evolution Logic ---

class EvolutionOperator:
    """
    Handles the evolutionary operations: selection, crossover, and mutation.
    """
    def __init__(self, candidate_pool_ids: List[int], mutation_rate: float = 0.1, survival_rate: float = 0.5):
        self.candidate_pool = candidate_pool_ids
        self.mutation_rate = mutation_rate
        self.survival_rate = survival_rate

    def crossover(self, parent1_cards: List[int], parent2_cards: List[int]) -> List[int]:
        """
        Performs crossover between two parent decks.
        Uses single-point crossover to maintain deck structure somewhat.
        """
        if len(parent1_cards) != len(parent2_cards):
             # Fallback for mismatched sizes: just take half from each
             size = len(parent1_cards)
             mid = size // 2
             return parent1_cards[:mid] + parent2_cards[mid : size - (size - mid) + mid]

        size = len(parent1_cards)
        if size < 2:
            return list(parent1_cards)

        point = random.randint(1, size - 1)
        child = parent1_cards[:point] + parent2_cards[point:]
        return child

    def mutate(self, cards: List[int]) -> List[int]:
        """
        Mutates the deck by randomly replacing cards.
        """
        new_cards = list(cards)
        if not self.candidate_pool:
            return new_cards

        for i in range(len(new_cards)):
            if random.random() < self.mutation_rate:
                new_cards[i] = random.choice(self.candidate_pool)
        return new_cards

    def evolve(self, current_pop: List[DeckIndividual], next_generation_index: int) -> List[DeckIndividual]:
        """
        Generates the next generation of decks.
        """
        # 1. Sort by fitness (win_rate)
        sorted_pop = sorted(current_pop, key=lambda d: d.win_rate, reverse=True)

        # 2. Select survivors (Elitism)
        survivor_count = max(2, int(len(current_pop) * self.survival_rate))
        survivors = sorted_pop[:survivor_count]

        # 3. Create next generation
        next_gen = []

        # Elitism: Survivors pass to next generation
        # We create new instances to reset stats for the new generation
        for s in survivors:
            # Keep the same ID to track lineage, or append gen?
            # Requirements usually imply tracking specific decks.
            # But stats reset per generation.
            # Let's keep the ID but reset stats in the new object.
            new_deck = DeckIndividual(s.deck_id, list(s.cards))
            next_gen.append(new_deck)

        # Fill the rest with offspring
        target_size = len(current_pop)
        child_index = 0

        while len(next_gen) < target_size:
            # Select parents from survivors
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)

            child_cards = self.crossover(p1.cards, p2.cards)
            child_cards = self.mutate(child_cards)

            # Generate ID
            new_id = f"gen{next_generation_index}_child{child_index:03d}"
            child_index += 1

            next_gen.append(DeckIndividual(new_id, child_cards))

        return next_gen

def run_evolution_loop(
    population_manager: PopulationManager,
    match_executor: ParallelMatchExecutor,
    evolution_operator: EvolutionOperator,
    generations: int,
    games_per_match: int = 10,
    threads_per_match: int = 1
):
    """
    Runs the evolutionary loop for a specified number of generations.
    """
    for gen in range(1, generations + 1):
        current_gen_index = population_manager.generation
        print(f"--- Starting Generation {current_gen_index} (Loop {gen}/{generations}) ---")
        current_pop = population_manager.get_population()

        if not current_pop:
            print("Population is empty. Aborting.")
            break

        # 1. Generate Matchups (Round Robin)
        matchups = []
        ids = [d.deck_id for d in current_pop]
        deck_map = {d.deck_id: d for d in current_pop}

        for id_a, id_b in itertools.combinations(ids, 2):
            matchups.append((id_a, deck_map[id_a].cards, id_b, deck_map[id_b].cards))

        # 2. Execute Matches
        print(f"Executing {len(matchups)} matchups (Round Robin)...")
        results = match_executor.execute_matchups(matchups, games_per_match, threads_per_match)

        # 3. Update Fitness
        wins = {id: 0 for id in ids}
        games = {id: 0 for id in ids}

        for res in results:
            id_a = res["deck_a_id"]
            id_b = res["deck_b_id"]
            w_a = res["wins_a"]
            w_b = res["wins_b"]
            draws = res["draws"]

            wins[id_a] += w_a
            games[id_a] += (w_a + w_b + draws)

            wins[id_b] += w_b
            games[id_b] += (w_a + w_b + draws)

            # Store specific matchup info (optional)
            if id_b not in deck_map[id_a].matchups:
                deck_map[id_a].matchups[id_b] = 0.0
            total_ab = w_a + w_b + draws
            if total_ab > 0:
                deck_map[id_a].matchups[id_b] = w_a / total_ab

            if id_a not in deck_map[id_b].matchups:
                 deck_map[id_b].matchups[id_a] = 0.0
            if total_ab > 0:
                 deck_map[id_b].matchups[id_a] = w_b / total_ab

        print("\nGeneration Results:")
        for d in current_pop:
            g = games[d.deck_id]
            if g > 0:
                d.win_rate = wins[d.deck_id] / g
            else:
                d.win_rate = 0.0
            d.games_played = g
            print(f"  Deck {d.deck_id}: WinRate {d.win_rate:.3f} ({wins[d.deck_id]}/{g})")

        # 4. Save Current Generation Stats
        stats_filename = f"gen{current_gen_index}_stats.json"
        population_manager.save_population(stats_filename)
        print(f"Saved stats to {stats_filename}")

        # 5. Evolve
        next_gen_decks = evolution_operator.evolve(current_pop, current_gen_index + 1)
        population_manager.update_generation(next_gen_decks)

        # 6. Save New Population
        population_manager.save_population("current_population.json")
        print(f"Generation {population_manager.generation} initialized and saved.\n")
