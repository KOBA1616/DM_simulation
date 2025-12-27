import json
import os
import sys
import glob

def check_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

    fail_count = 0

    def check_card(card, index_info=""):
        if 'actions' in card:
            print(f"FAILURE: 'actions' field found in {filepath} at {index_info} (ID: {card.get('id', 'Unknown')})")
            return 1
        return 0

    if isinstance(data, list):
        for i, card in enumerate(data):
            fail_count += check_card(card, f"item {i}")
    elif isinstance(data, dict):
        if 'cards' in data:
             for i, card in enumerate(data['cards']):
                 fail_count += check_card(card, f"cards[{i}]")
        else:
            if 'id' in data: # Single card
                fail_count += check_card(data)
            else:
                 for k, v in data.items():
                     if isinstance(v, dict):
                         fail_count += check_card(v, f"key {k}")

    return fail_count

def main():
    target_dir = "data"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]

    files = glob.glob(os.path.join(target_dir, "**/*.json"), recursive=True)

    total_fails = 0
    for f in files:
        if "unmapped_actions.json" in f: continue
        total_fails += check_file(f)

    if total_fails > 0:
        print(f"FAILED: Found {total_fails} instances of legacy 'actions' usage.")
        sys.exit(1)
    else:
        print("SUCCESS: No legacy 'actions' fields found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
