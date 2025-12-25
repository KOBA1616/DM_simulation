import json
import os
import sys
import random

# Ensure proper path for running inside the repo
# We must add the path BEFORE importing the module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure it is built and in the bin/ directory.")
    sys.exit(1)

def verify_deck_evolution_logic() -> None:
    print("Verifying Deck Evolution Logic (C++ Module)...")

    # 1. Load Real Card Data
    json_path = "data/cards.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    # Load using JsonLoader. This returns a dict {id: CardDefinition} in Python via pybind11 map conversion
    card_db = dm_ai_module.JsonLoader.load_cards(json_path)

    print(f"Loaded {len(card_db)} cards into database.")
    if len(card_db) == 0:
        print("Error: No cards loaded.")
        return

    # 2. Setup Configuration
    config = dm_ai_module.DeckEvolutionConfig()
    config.target_deck_size = 40
    config.mutation_rate = 0.5  # High rate to ensure change

    # 3. Instantiate C++ DeckEvolution
    evolver = dm_ai_module.DeckEvolution(card_db)

    # 4. Prepare Test Data
    # Get available IDs
    available_ids = list(card_db.keys())

    # Create initial deck (random selection of 40)
    current_deck = []
    for _ in range(40):
        current_deck.append(random.choice(available_ids))

    # Create candidate pool (all available cards repeated)
    candidate_pool = []
    for _ in range(200):
         candidate_pool.append(random.choice(available_ids))

    print(f"Initial Deck Size: {len(current_deck)}")
    print(f"Candidate Pool Size: {len(candidate_pool)}")

    # 5. Run Evolution
    print("Running evolve_deck...")
    new_deck = evolver.evolve_deck(current_deck, candidate_pool, config)

    print(f"New Deck Size: {len(new_deck)}")

    # Verify deck size
    assert len(new_deck) == 40, f"Expected deck size 40, got {len(new_deck)}"

    # Verify deck changed
    if current_deck != new_deck:
        print("Evolution Verification Passed: Deck was mutated.")
    else:
        print("Warning: Deck did not change. This might be chance, but with mutation_rate 0.5 it is unlikely.")

    # 6. Verify Interaction Score
    print("Calculating Interaction Score...")
    score = evolver.calculate_interaction_score(new_deck)
    print(f"Interaction Score: {score}")

    # 7. Verify Candidates by Civ
    # Find a civilization that exists in the DB
    test_civ = dm_ai_module.Civilization.FIRE

    print("Fetching candidates for Fire civilization...")
    fire_candidates = evolver.get_candidates_by_civ(candidate_pool, test_civ)
    print(f"Found {len(fire_candidates)} Fire candidates.")

    # Verify filters work
    for cid in fire_candidates:
        # Check if the card has the civ
        # Note: card_db[cid].civilizations is a list of enums
        has_civ = False
        for c in card_db[cid].civilizations:
            if c == test_civ:
                has_civ = True
                break
        assert has_civ, f"Card {cid} was returned but does not have Fire civilization."

    print("Deck Evolution C++ Integration Verification Passed.")

if __name__ == "__main__":
    verify_deck_evolution_logic()
