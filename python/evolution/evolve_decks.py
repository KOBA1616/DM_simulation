
import os
import sys
import json
import random
import argparse
from dataclasses import asdict

# Setup Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module.")
    sys.exit(1)

def evolve_decks_cli(meta_deck_path: str, new_deck_path: str, card_db_path: str):
    """
    Evolves decks based on current meta.
    Simple implementation:
    1. Loads meta decks.
    2. Evolves one of them (or a random one).
    3. Saves the new deck to new_deck_path (and optionally updates meta).
    """

    # Load Cards
    card_db = dm_ai_module.JsonLoader.load_cards(card_db_path)

    # Load Meta Decks
    meta_decks = []
    if os.path.exists(meta_deck_path):
        with open(meta_deck_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    meta_decks = data
                elif isinstance(data, dict) and "decks" in data:
                    meta_decks = data["decks"]
            except:
                pass

    if not meta_decks:
        print("No meta decks found. Creating random deck.")
        # Create random deck
        all_ids = list(card_db.keys())
        base_deck = [random.choice(all_ids) for _ in range(40)]
    else:
        # Pick one to evolve
        # Format of meta_decks usually list of list of ints? Or objects?
        # Assuming list of list of ints for simplicity or list of dicts.
        # Let's assume list of list of ints.
        base_obj = random.choice(meta_decks)
        if isinstance(base_obj, list):
            base_deck = base_obj
        elif isinstance(base_obj, dict) and "cards" in base_obj:
            base_deck = base_obj["cards"]
        else:
            all_ids = list(card_db.keys())
            base_deck = [random.choice(all_ids) for _ in range(40)]

    # Config
    config = dm_ai_module.DeckEvolutionConfig()
    config.target_deck_size = 40
    config.mutation_rate = 0.1

    evolver = dm_ai_module.DeckEvolution(card_db)

    # Candidate Pool: All cards for now
    # Ideally should be filtered by civ or availability
    candidate_pool = list(card_db.keys()) * 5 # Allow duplicates

    print(f"Evolving deck with size {len(base_deck)}...")
    new_deck_cards = evolver.evolve_deck(base_deck, candidate_pool, config)

    # Save
    with open(new_deck_path, 'w', encoding='utf-8') as f:
        json.dump(new_deck_cards, f, ensure_ascii=False)

    print(f"Evolved deck saved to {new_deck_path}")

    # Update Meta?
    # PBT Controller will decide whether to add it to meta_decks.json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_decks", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cards", type=str, default="data/cards.json")

    args = parser.parse_args()
    evolve_decks_cli(args.meta_decks, args.output, args.cards)
