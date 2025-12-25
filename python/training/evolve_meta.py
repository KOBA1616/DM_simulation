
import sys
import os
import random
import json
import time

# Ensure dm_ai_module is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
try:
    import dm_ai_module
except ImportError:
    print("Error: dm_ai_module not found. Please build the project first.")
    sys.exit(1)

def load_card_db():
    json_loader = dm_ai_module.JsonLoader()
    card_db = json_loader.load_cards("data/cards.json")
    return card_db

def create_starter_decks(card_db, count=10):
    all_ids = list(card_db.keys())
    decks = []

    # Simple heuristic
    civ_groups = {
        dm_ai_module.Civilization.FIRE: [],
        dm_ai_module.Civilization.WATER: [],
        dm_ai_module.Civilization.NATURE: [],
        dm_ai_module.Civilization.LIGHT: [],
        dm_ai_module.Civilization.DARKNESS: []
    }

    for cid in all_ids:
        c = card_db[cid]
        for civ in c.civilizations:
            civ_groups[civ].append(cid)

    for i in range(count):
        deck = []
        civs = random.sample(list(civ_groups.keys()), k=random.randint(1, 2))
        pool = []
        for civ in civs:
            pool.extend(civ_groups[civ])
        if not pool: pool = all_ids

        while len(deck) < 40:
            cid = random.choice(pool)
            if deck.count(cid) < 4:
                deck.append(cid)
        decks.append(deck)

    return decks

def run_evolution_loop(generations=1, matches_per_pair=1):
    print("Loading Card DB...")
    card_db = load_card_db()

    print("Initializing MetaEnvironment...")
    meta_env = dm_ai_module.MetaEnvironment(card_db)

    # Very small population for testing
    pop_size = 4
    print(f"Creating {pop_size} starter decks...")
    initial_decks = create_starter_decks(card_db, count=pop_size)
    meta_env.initialize_population(initial_decks)

    # 4th arg is num_threads, not sims.
    runner = dm_ai_module.ParallelRunner(card_db, 1, 1)

    all_card_ids = list(card_db.keys())

    for gen in range(generations):
        print(f"\n--- Generation {gen} ---")
        population = meta_env.get_population()

        pairings = []
        for i in range(len(population)):
            opp_idx = (i + 1) % len(population)
            pairings.append((i, opp_idx))

        print(f"Running {len(pairings)} matches...")

        for i, (p1_idx, p2_idx) in enumerate(pairings):
            deck1 = population[p1_idx].deck
            deck2 = population[p2_idx].deck

            # play_deck_matchup(deck1, deck2, num_games, num_threads)
            results = runner.play_deck_matchup(deck1, deck2, 1, 1)
            winner = results[0]

            meta_env.record_match(p1_idx, p2_idx, winner)
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(pairings)} matches")

        # Step Generation
        print("Evolving next generation...")
        meta_env.step_generation(all_card_ids)

        population = meta_env.get_population()
        sorted_pop = sorted(population, key=lambda a: a.elo_rating, reverse=True)
        top = sorted_pop[0]
        print(f"Best Deck (Gen {top.generation}, Elo {top.elo_rating:.1f}): Wins {top.wins}/{top.matches_played}")

    print("\nEvolution Complete.")

if __name__ == "__main__":
    # 1 generation, minimal config
    run_evolution_loop(generations=1)
