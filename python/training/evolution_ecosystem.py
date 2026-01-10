import json
import os
import time
import multiprocessing
from typing import List, Dict, Tuple, Optional, Any

try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

# --- Data Structures ---

class DeckIndividual:
    """Represents a single deck in the population."""
    def __init__(self, deck_id: str, card_ids: List[int]):
        self.deck_id = deck_id
        self.cards = card_ids
        self.fitness = 0.0
        self.wins = 0
        self.games = 0
        self.generation = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deck_id": self.deck_id,
            "cards": self.cards,
            "fitness": self.fitness,
            "wins": self.wins,
            "games": self.games,
            "generation": self.generation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeckIndividual':
        ind = cls(data["deck_id"], data["cards"])
        ind.fitness = data.get("fitness", 0.0)
        ind.wins = data.get("wins", 0)
        ind.games = data.get("games", 0)
        ind.generation = data.get("generation", 0)
        return ind

# --- Population Manager ---

class PopulationManager:
    """
    Manages the population of decks for PBT.
    Handles initialization, saving/loading, and basic pool management.
    """
    def __init__(self, population_size: int = 20, storage_dir: str = "data/evolution"):
        self.population_size = population_size
        self.storage_dir = storage_dir
        self.population: List[DeckIndividual] = []

        os.makedirs(self.storage_dir, exist_ok=True)

    def initialize_random_population(self, card_pool: List[int], deck_size: int = 40):
        """Generates random initial population."""
        import random
        self.population = []
        for i in range(self.population_size):
            # Simple random deck generation
            deck = [random.choice(card_pool) for _ in range(deck_size)]
            ind = DeckIndividual(f"gen0_deck{i:03d}", deck)
            self.population.append(ind)
        print(f"[PopulationManager] Initialized {len(self.population)} random decks.")

    def save_population(self, generation: int):
        """Saves current population to disk."""
        filename = os.path.join(self.storage_dir, f"population_gen{generation:04d}.json")
        data = {
            "generation": generation,
            "timestamp": time.time(),
            "individuals": [ind.to_dict() for ind in self.population]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"[PopulationManager] Saved generation {generation} to {filename}")

    def load_population(self, generation: int):
        """Loads population from disk."""
        filename = os.path.join(self.storage_dir, f"population_gen{generation:04d}.json")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Population file not found: {filename}")

        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.population = [DeckIndividual.from_dict(d) for d in data["individuals"]]
        print(f"[PopulationManager] Loaded {len(self.population)} decks from generation {generation}.")

    def get_matchup_pairs(self) -> List[Tuple[str, List[int], str, List[int]]]:
        """
        Generates matchup pairs for evaluation.
        Currently implements simple round-robin or random pairing logic.
        """
        import random
        pairs = []
        # Simple random pairing for self-play
        # Each deck plays against 2 random opponents
        for ind in self.population:
             opponents = random.sample(self.population, 2)
             for opp in opponents:
                 if ind.deck_id != opp.deck_id:
                     pairs.append((ind.deck_id, ind.cards, opp.deck_id, opp.cards))
        return pairs

    def update_fitness(self, results: Dict[str, Dict[str, int]]):
        """
        Updates fitness/stats based on match results.
        results: {deck_id: {'wins': w, 'games': g}}
        """
        for ind in self.population:
            if ind.deck_id in results:
                res = results[ind.deck_id]
                ind.wins += res.get('wins', 0)
                ind.games += res.get('games', 0)
                if ind.games > 0:
                    ind.fitness = ind.wins / ind.games
                else:
                    ind.fitness = 0.0

# --- Parallel Match Executor ---

def worker_play_batch(matchup_batch, cards_json_path, games_per_match=2, sims=100):
    """
    Worker function to execute a batch of matches using C++ ParallelRunner.
    Must be top-level for multiprocessing.
    """
    if not dm_ai_module:
        return []

    try:
        # Load cards per worker to avoid pickling C++ objects
        card_db = dm_ai_module.JsonLoader.load_cards(cards_json_path)

        # ParallelRunner expects map<int, CardDefinition> (card_db)
        runner = dm_ai_module.ParallelRunner(card_db, sims, 1)

    except Exception as e:
        print(f"[Worker] Failed to init ParallelRunner: {e}")
        return []

    batch_results = []

    for p1_deck_id, p1_cards, p2_deck_id, p2_cards in matchup_batch:
        try:
            stats = runner.play_deck_matchup(p1_cards, p2_cards, games_per_match, 1)

            p1_wins = stats.get(1, 0)
            p2_wins = stats.get(2, 0)
            draws = stats.get(3, 0)

            batch_results.append({
                "p1_id": p1_deck_id,
                "p2_id": p2_deck_id,
                "p1_wins": p1_wins,
                "p2_wins": p2_wins,
                "draws": draws,
                "total": games_per_match
            })

        except Exception as e:
            print(f"[Worker] Error in match execution: {e}")

    return batch_results

class ParallelMatchExecutor:
    """
    Executes matches in parallel using multiprocessing.
    """
    def __init__(self, cards_json_path: str, num_workers: int = 4):
        self.cards_json_path = cards_json_path
        self.num_workers = num_workers
        # We don't load card_db here to avoid passing it to workers
        self.bin_path = "bin/" # Dummy for test compat

    def execute_matchups(self, matchups: List[Tuple[str, List[int], str, List[int]]],
                         games_per_match: int = 10, threads_per_match: int = 1) -> Dict[str, Dict[str, int]]:
        """
        Runs the matchups in parallel.
        Returns aggregated stats per deck.
        """
        if not matchups:
            return {}

        chunk_size = max(1, len(matchups) // self.num_workers)
        chunks = [matchups[i:i + chunk_size] for i in range(0, len(matchups), chunk_size)]

        # Pass path instead of DB object
        args_list = [(chunk, self.cards_json_path, games_per_match, 100) for chunk in chunks]

        results_agg = {}

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results_nested = pool.starmap(worker_play_batch, args_list)

            for batch_res in results_nested:
                for res in batch_res:
                    p1 = res['p1_id']
                    p2 = res['p2_id']

                    if p1 not in results_agg: results_agg[p1] = {'wins': 0, 'games': 0}
                    if p2 not in results_agg: results_agg[p2] = {'wins': 0, 'games': 0}

                    results_agg[p1]['wins'] += res['p1_wins']
                    results_agg[p1]['games'] += res['total']

                    results_agg[p2]['wins'] += res['p2_wins']
                    results_agg[p2]['games'] += res['total']

        return results_agg

# --- Evolution Operator (Task 3.1) ---

class EvolutionOperator:
    """Handles selection, crossover, and mutation."""
    def __init__(self, card_pool: List[int], selection_rate: float = 0.5, mutation_rate: float = 0.1):
        self.card_pool = card_pool
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate

    def evolve_population(self, population: List[DeckIndividual]) -> List[DeckIndividual]:
        import random
        # 1. Selection
        # Sort by fitness desc
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        survivor_count = max(2, int(len(population) * self.selection_rate))
        survivors = sorted_pop[:survivor_count]

        next_gen = []
        next_gen.extend(survivors) # Elitism

        # 2. Crossover & Mutation to fill rest
        while len(next_gen) < len(population):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)

            child_cards = self._crossover(parent1.cards, parent2.cards)
            child_cards = self._mutate(child_cards)

            child = DeckIndividual(f"gen{parent1.generation + 1}_child{len(next_gen)}", child_cards)
            child.generation = parent1.generation + 1
            next_gen.append(child)

        return next_gen

    def _crossover(self, deck1: List[int], deck2: List[int]) -> List[int]:
        # Single point crossover
        split = len(deck1) // 2
        child = deck1[:split] + deck2[split:]
        return child[:40] # Ensure size

    def _mutate(self, deck: List[int]) -> List[int]:
        import random
        # Randomly replace cards
        new_deck = list(deck)
        if random.random() < self.mutation_rate:
             # Number of mutations
             num_mutations = random.randint(1, 4)
             for _ in range(num_mutations):
                 idx = random.randint(0, len(new_deck) - 1)
                 new_card = random.choice(self.card_pool)
                 new_deck[idx] = new_card
        return new_deck

def run_evolution_loop(generations: int = 10, pop_size: int = 20):
    """Main loop for PBT."""
    # Setup
    manager = PopulationManager(population_size=pop_size)

    try:
        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        card_pool = list(card_db.keys())
    except:
        print("Failed to load cards.json")
        return

    manager.initialize_random_population(card_pool)

    executor = ParallelMatchExecutor("data/cards.json")
    operator = EvolutionOperator(card_pool)

    for gen in range(generations):
        print(f"--- Generation {gen} ---")

        # 1. Matchups
        pairs = manager.get_matchup_pairs()

        # 2. Execute
        results = executor.execute_matchups(pairs)

        # 3. Update
        manager.update_fitness(results)
        manager.save_population(gen)

        # 4. Evolve
        manager.population = operator.evolve_population(manager.population)

if __name__ == "__main__":
    run_evolution_loop(generations=1, pop_size=4)
