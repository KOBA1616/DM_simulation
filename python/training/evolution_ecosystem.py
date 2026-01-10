
import os
import json
import random
import copy
import multiprocessing
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

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
            # Respect 4-copy rule
            current_counts = Counter()
            while len(deck_cards) < deck_size:
                card_id = random.choice(candidate_pool_ids)
                if current_counts[card_id] < 4:
                    deck_cards.append(card_id)
                    current_counts[card_id] += 1

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

class EvolutionOperator:
    """
    Handles the genetic operations: Selection, Crossover, and Mutation.
    """
    def __init__(self, candidate_pool_ids: List[int], mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.candidate_pool_ids = candidate_pool_ids
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def select_survivors(self, population: List[DeckIndividual], selection_rate: float = 0.5) -> List[DeckIndividual]:
        """
        Selects the top performing decks based on win rate.
        """
        # Sort by win rate descending
        sorted_pop = sorted(population, key=lambda d: d.win_rate, reverse=True)
        cutoff = int(len(population) * selection_rate)
        # Ensure at least 2 survivors if population >= 2
        if cutoff < 2 and len(population) >= 2:
            cutoff = 2
        return sorted_pop[:cutoff]

    def crossover(self, parent1_cards: List[int], parent2_cards: List[int]) -> List[int]:
        """
        Creates a child deck from two parent decks.
        Uses a pool-based approach to respect the 4-copy limit.
        """
        deck_size = len(parent1_cards)
        child_cards = []
        current_counts = Counter()

        # Combine parents
        # Strategy: Take roughly half from each, but ensure validity
        # Simple implementation: Shuffle both parents together, then pick cards until full
        # To preserve "genes", we should try to keep structure.
        # Let's use Uniform Crossover: for each slot, pick from P1 or P2

        # Note: Deck order doesn't matter for gameplay (it's shuffled),
        # but for crossover, index-based mapping might be arbitrary.
        # Better: construct a pool of (card, count) from parents?

        # Implementation:
        # 1. Take 50% of cards from Parent 1
        # 2. Fill rest from Parent 2
        # 3. Fill rest from Candidate Pool (if needed to satisfy constraints)

        # Let's try: Take random 50% subset from P1.
        p1_subset = random.sample(parent1_cards, k=deck_size // 2)
        for card in p1_subset:
            if current_counts[card] < 4:
                child_cards.append(card)
                current_counts[card] += 1

        # Fill from P2
        # We shuffle P2 to get random genes
        p2_shuffled = list(parent2_cards)
        random.shuffle(p2_shuffled)

        for card in p2_shuffled:
            if len(child_cards) >= deck_size:
                break
            if current_counts[card] < 4:
                child_cards.append(card)
                current_counts[card] += 1

        # If still not full (due to 4-copy constraint rejecting P2 cards), fill from pool
        while len(child_cards) < deck_size:
            card = random.choice(self.candidate_pool_ids)
            if current_counts[card] < 4:
                child_cards.append(card)
                current_counts[card] += 1

        return child_cards

    def mutate(self, deck_cards: List[int]) -> List[int]:
        """
        Mutates a deck by replacing random cards.
        """
        deck_size = len(deck_cards)
        mutated_cards = list(deck_cards)
        num_mutations = int(deck_size * self.mutation_rate)

        if num_mutations == 0 and random.random() < self.mutation_rate:
            num_mutations = 1

        # Remove random cards
        for _ in range(num_mutations):
            if not mutated_cards:
                break
            idx = random.randrange(len(mutated_cards))
            mutated_cards.pop(idx)

        # Re-fill with random cards from pool
        current_counts = Counter(mutated_cards)
        while len(mutated_cards) < deck_size:
            card = random.choice(self.candidate_pool_ids)
            if current_counts[card] < 4:
                mutated_cards.append(card)
                current_counts[card] += 1

        return mutated_cards

    def create_next_generation(self, current_pop: List[DeckIndividual], generation_idx: int, target_size: int) -> List[DeckIndividual]:
        """
        Generates the next generation of decks.
        """
        survivors = self.select_survivors(current_pop)
        next_gen = []

        # Elitism: Keep survivors (or top 1?)
        # Let's keep top 1 as is, and others as parents
        # Actually, requirement says "Selection (top 50%)". Usually means these are the parents.
        # We can keep them in the next gen (Elitism) to ensure we don't regress.
        # Let's keep all survivors.
        for s in survivors:
            # Create a copy with new ID or keep same?
            # If we keep same ID, we track history.
            # But let's create new ID to denote it survived to this gen.
            # Or just deepcopy.
            survivor_deck = DeckIndividual(s.deck_id, list(s.cards), s.win_rate) # Reset win_rate? Usually yes for new eval.
            survivor_deck.win_rate = 0.0
            survivor_deck.games_played = 0
            survivor_deck.matchups = {}
            next_gen.append(survivor_deck)

        # Fill the rest
        deck_idx = 0
        while len(next_gen) < target_size:
            if not survivors:
                # Should not happen if population initialized
                break

            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)

            child_cards = self.crossover(parent1.cards, parent2.cards)
            child_cards = self.mutate(child_cards)

            new_id = f"gen{generation_idx}_deck{deck_idx:03d}"
            deck_idx += 1

            # Ensure ID is unique if colliding (unlikely with gen prefix)
            next_gen.append(DeckIndividual(new_id, child_cards))

        return next_gen[:target_size]

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
        self.bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bin'))

    def execute_matchups(
        self,
        matchups: List[Tuple[str, List[int], str, List[int]]],
        games_per_match: int = 100,
        threads_per_match: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Executes a list of matchups in parallel.
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
            results_nested = pool.starmap(worker_play_batch, args_list)

        all_results = [res for sublist in results_nested for res in sublist]

        elapsed = time.time() - start_time
        print(f"[ParallelMatchExecutor] Completed in {elapsed:.2f}s ({len(all_results)} matchups)")

        return all_results

def run_evolution_loop(
    card_db_path: str,
    population_size: int = 20,
    generations: int = 100,
    games_per_match: int = 10,
    num_workers: int = 4,
    storage_path: str = "data/population"
):
    """
    Main loop for the evolution ecosystem.
    """
    # 1. Initialize
    print("Initializing Evolution Ecosystem...")

    # Load Card DB (for ID checks and candidate pool)
    # We need dm_ai_module here to load cards, or we parse JSON manually if dm_ai_module not available?
    # The script likely runs where dm_ai_module is available.

    if dm_ai_module:
        card_db = dm_ai_module.JsonLoader.load_cards(card_db_path)
        # Convert to python dict if it's not already (it is usually map-like)
        # We only need IDs for candidate pool.
        # But wait, JsonLoader returns a map.
        # Let's assume keys are IDs.
        # Note: In C++ binding, card_db might be a dict-like object.
        # Let's iterate keys.
        # If card_db is not iterable in python, we might need a helper.
        # But based on memory, it acts like a dict.
        candidate_pool_ids = []
        # card_db is likely {id: CardDefinition}
        # We need to know which cards are valid for deck building (e.g. not tokens).
        # For now, let's assume all loaded cards are valid.
        # Or better: filter by card type or metadata if possible.
        # Let's just take all IDs.
        # Note: If it's a C++ map, keys() might work.
        try:
             # Try to get list of IDs.
             # If it supports iteration:
             candidate_pool_ids = list(card_db.keys())
        except:
             # Fallback: maybe it doesn't support keys().
             # Using ranges?
             # For now, let's assume we can get IDs.
             pass
    else:
        # Fallback for testing/stub
        print("Warning: dm_ai_module not found. Using dummy pool.")
        card_db = {}
        candidate_pool_ids = list(range(1, 100)) # Dummy IDs

    if not candidate_pool_ids:
        print("Error: No candidates found for deck generation.")
        return

    pop_manager = PopulationManager(card_db, population_size, storage_path)
    # Check if we have saved population
    try:
        pop_manager.load_population()
        print(f"Loaded existing population: Generation {pop_manager.generation}")
    except FileNotFoundError:
        print("Creating new random population...")
        pop_manager.initialize_random_population(candidate_pool_ids)
        pop_manager.save_population()

    executor = ParallelMatchExecutor(card_db_path, num_workers)
    evolution_op = EvolutionOperator(candidate_pool_ids)

    # 2. Evolution Loop
    for gen in range(pop_manager.generation, generations):
        print(f"--- Generation {gen} ---")

        current_pop = pop_manager.get_population()

        # A. Generate Matchups (Round Robin)
        matchups = []
        for i in range(len(current_pop)):
            for j in range(i + 1, len(current_pop)):
                d1 = current_pop[i]
                d2 = current_pop[j]
                matchups.append((d1.deck_id, d1.cards, d2.deck_id, d2.cards))

        # B. Execute Matches
        print(f"Running {len(matchups)} matchups...")
        results = executor.execute_matchups(matchups, games_per_match=games_per_match)

        # C. Update Win Rates
        # Reset stats
        for deck in current_pop:
            deck.win_rate = 0.0
            deck.games_played = 0
            deck.matchups = {}

        # Accumulate results
        deck_wins = {d.deck_id: 0 for d in current_pop}
        deck_games = {d.deck_id: 0 for d in current_pop}

        for res in results:
            id_a = res["deck_a_id"]
            id_b = res["deck_b_id"]
            wins_a = res["wins_a"]
            wins_b = res["wins_b"]
            draws = res["draws"]
            total = wins_a + wins_b + draws

            deck_wins[id_a] += wins_a
            deck_games[id_a] += total

            deck_wins[id_b] += wins_b
            deck_games[id_b] += total

            # Store matchup details? (Optional)

        # Calculate rates
        for deck in current_pop:
            if deck_games[deck.deck_id] > 0:
                deck.win_rate = deck_wins[deck.deck_id] / deck_games[deck.deck_id]
                deck.games_played = deck_games[deck.deck_id]

        # Log Top Decks
        current_pop.sort(key=lambda d: d.win_rate, reverse=True)
        top_deck = current_pop[0]
        print(f"Generation {gen} Top Deck: {top_deck.deck_id} (WR: {top_deck.win_rate:.2f})")

        # D. Evolve
        next_gen_decks = evolution_op.create_next_generation(current_pop, gen + 1, population_size)

        # E. Update and Save
        pop_manager.update_generation(next_gen_decks)
        pop_manager.save_population()
        pop_manager.save_population(f"gen{gen}_population.json") # Snapshot

    print("Evolution completed.")

if __name__ == "__main__":
    # Example usage
    # Ensure this runs from repo root or set paths correctly
    card_db_path = "data/cards.json"
    if os.path.exists(card_db_path):
        run_evolution_loop(card_db_path, population_size=4, generations=2, games_per_match=2)
    else:
        print(f"Card DB not found at {card_db_path}")
