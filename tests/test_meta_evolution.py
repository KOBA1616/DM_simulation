import copy
import random
from typing import List, Tuple


def deterministic_score(deck: List[int]) -> int:
    """Simple deterministic score for a deck: sum of ids modulo a constant.

    This is intentionally trivial but deterministic for fixed deck composition.
    """
    return sum(deck) % 1000


def mutate_deck(deck: List[int], pool: List[int], mutation_rate: float, rng: random.Random) -> List[int]:
    new = deck.copy()
    for i in range(len(new)):
        if rng.random() < mutation_rate:
            new[i] = rng.choice(pool)
    return new


def evaluate_population(pop: List[List[int]]) -> List[Tuple[List[int], float]]:
    """Score each deck and return list of (deck, score) sorted desc."""
    scored = [(d, deterministic_score(d)) for d in pop]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in scored]


def run_minimal_evolution(seed: int = 1234,
                          pool: List[int] = None,
                          init_decks: int = 4,
                          generations: int = 2,
                          mutation_rate: float = 0.2) -> List[List[int]]:
    rng = random.Random(seed)

    if pool is None:
        # minimal pool if cards.json not available
        pool = list(range(1, 21))

    # Initialize population: random decks of size 40
    population: List[List[int]] = []
    for _ in range(init_decks):
        deck = [rng.choice(pool) for _ in range(40)]
        population.append(deck)

    for gen in range(generations):
        # Mutate each deck to create challengers
        challengers: List[List[int]] = []
        for deck in population:
            challenger = mutate_deck(deck, pool, mutation_rate, rng)
            challengers.append(challenger)

        # Combine and evaluate
        combined = population + challengers
        ranked = evaluate_population(combined)

        # Select top `init_decks` as next generation
        population = [copy.deepcopy(d) for d, _ in ranked[:init_decks]]

    return population


def test_meta_evolution_reproducible():
    """Run the minimal evolution twice with fixed seed and assert reproducible output."""
    seed = 20260312
    pop1 = run_minimal_evolution(seed=seed, pool=list(range(1, 50)), init_decks=4, generations=2, mutation_rate=0.15)
    pop2 = run_minimal_evolution(seed=seed, pool=list(range(1, 50)), init_decks=4, generations=2, mutation_rate=0.15)

    assert len(pop1) == 4 and len(pop2) == 4
    # Deep equality ensures run is deterministic for fixed seed
    assert pop1 == pop2


def test_meta_evolution_changes_with_different_seed():
    a = run_minimal_evolution(seed=1, pool=list(range(1, 50)), init_decks=4, generations=2)
    b = run_minimal_evolution(seed=2, pool=list(range(1, 50)), init_decks=4, generations=2)
    # Very unlikely to be identical with different seeds
    assert a != b
