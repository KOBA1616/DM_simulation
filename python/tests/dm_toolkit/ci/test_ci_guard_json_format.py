
import json
import os
import pytest
import glob

# Ensure data path relative to repo root
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")

def get_json_files():
    # Recursively find all JSON files in data directory
    # Note: excluding 'test_*.json' might be appropriate if they are explicitly legacy
    # But for now, we want strict enforcement unless exception is made.
    return glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True)

@pytest.mark.parametrize("filepath", get_json_files())
def test_json_no_actions_key(filepath):
    """
    CI Guard: Ensure that JSON files in data/ do not contain the legacy 'actions' key.
    The system should use 'commands' instead.
    """

    # Exceptions: List files that are allowed to have 'actions' (e.g., legacy test data)
    # Adjust this list as needed.
    EXCEPTIONS = [
        "dummy_draw.json", # Created by test_command_expansion.py
        "dummy_destroy.json", # Created by test_command_expansion.py
        # Add other exceptions if necessary, e.g. "test_cards.json" if it hasn't been migrated yet.
    ]

    filename = os.path.basename(filepath)
    if filename in EXCEPTIONS:
        pytest.skip(f"Skipping exception file: {filename}")

    # Also ignore hidden files or stats files that are not card definitions
    if filename.startswith(".") or "stats" in filename or "scenarios" in filename or "decks" in filename:
        # Assuming stats/scenarios/decks don't use 'actions' in a way that conflicts, or have different schema.
        # But 'scenarios.json' might embed card overrides?
        # Let's check card structure specifically.
        pass

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        pytest.fail(f"Invalid JSON file: {filepath}")

    def check_object(obj, path=""):
        if isinstance(obj, dict):
            # Check for 'actions' key
            if "actions" in obj:
                # Confirm context: is it inside an effect?
                # Heuristic: if 'trigger' is present or it's inside 'effects' list
                pytest.fail(f"Found 'actions' key in {filepath} at {path}. Legacy actions are forbidden.")

            for k, v in obj.items():
                check_object(v, path + f".{k}")

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_object(item, path + f"[{i}]")

    # Only validate if it looks like a card definition file (list of objects with 'id', 'type')
    # or a single card object.
    is_card_file = False
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "id" in data[0]:
        is_card_file = True
    elif isinstance(data, dict) and "id" in data and "type" in data:
        is_card_file = True
    elif isinstance(data, dict) and "cards" in data: # Some formats might wrap it
        is_card_file = True

    if is_card_file:
        check_object(data)
