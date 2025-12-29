import json
import os
import pytest

# Adjust path to be relative to this script or project root
# Assuming tests/verify_json_integrity.py is run from root or tests/
# We will try to find data/cards.json relative to the script location.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CARDS_JSON_PATH = os.path.join(PROJECT_ROOT, 'data', 'cards.json')

def test_json_integrity():
    """
    Verifies that cards.json does not contain legacy 'actions' fields
    and strictly uses 'commands'.
    """
    if not os.path.exists(CARDS_JSON_PATH):
        pytest.fail(f"cards.json not found at {CARDS_JSON_PATH}")

    with open(CARDS_JSON_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    errors = []

    # cards.json is a list of cards or a dict of id->card?
    # Usually it's a list in this project, but let's handle both or check memory/file.
    # Memory says "dm_ai_module.JsonLoader.load_cards"
    # Inspecting structure: standard is usually a list of objects.

    iterable = cards
    if isinstance(cards, dict):
        iterable = cards.values()

    for card in iterable:
        card_id = card.get('id', 'Unknown')
        name = card.get('name', 'Unknown')

        # Check top-level
        if 'actions' in card:
             errors.append(f"Card {card_id} ({name}): Found legacy 'actions' at root.")

        # Check effects
        if 'effects' in card:
            for idx, effect in enumerate(card['effects']):
                if 'actions' in effect:
                    errors.append(f"Card {card_id} ({name}) Effect #{idx}: Found legacy 'actions'.")

    if errors:
        pytest.fail("\n".join(errors))

if __name__ == "__main__":
    try:
        test_json_integrity()
        print("Integrity check passed: No legacy 'actions' found.")
    except Exception as e:
        print(f"Integrity check failed: {e}")
        exit(1)
