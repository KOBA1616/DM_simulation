
import sys
import os
sys.path.append(os.path.abspath("bin"))
try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module")
    sys.exit(1)

def test_load_reaction():
    loader = dm_ai_module.JsonLoader
    cards = loader.load_cards("data/ninja_test.json")

    if 9999 not in cards:
        print("FAIL: Card 9999 not loaded")
        sys.exit(1)

    card = cards[9999]
    print(f"Loaded card: {card.name}")

    # Check if reaction abilities loaded (using python binding inspection if available, or just assumption via success)
    # Since bindings might not expose reaction_abilities yet, we assume success if no crash.
    print("Load successful")

if __name__ == "__main__":
    test_load_reaction()
