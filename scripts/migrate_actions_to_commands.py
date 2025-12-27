import json
import os
import sys
import glob

# Ensure project root is in path
sys.path.append(os.getcwd())

from dm_toolkit.action_mapper import ActionToCommandMapper

def migrate_file(filepath):
    print(f"Migrating {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    modified = False

    def process_card(card):
        nonlocal modified
        if 'actions' in card:
            # Check if already has commands
            if 'commands' not in card:
                card['commands'] = []

            # Convert actions to commands
            # legacy actions are usually a list of dicts
            actions = card.get('actions', [])
            if isinstance(actions, list):
                for act in actions:
                    cmd = ActionToCommandMapper.map_action(act)
                    card['commands'].append(cmd)

            # Remove actions
            del card['actions']
            modified = True

        # Recursive check for nested structures if any?
        # Typically cards are flat list in cards.json
        pass

    if isinstance(data, list):
        for card in data:
            process_card(card)
    elif isinstance(data, dict):
        # Maybe a wrapper or dict of cards
        if 'cards' in data:
             for card in data['cards']:
                 process_card(card)
        else:
            # Single card or map? assume single card if has id
            if 'id' in data:
                process_card(data)
            else:
                 # Dictionary of id -> card?
                 for k, v in data.items():
                     if isinstance(v, dict):
                         process_card(v)

    if modified:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Updated {filepath}")
            return True
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False
    else:
        print(f"No changes needed for {filepath}")
        return False

def main():
    target_dir = "data"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]

    # Find json files
    files = glob.glob(os.path.join(target_dir, "**/*.json"), recursive=True)

    count = 0
    for f in files:
        if "unmapped_actions.json" in f: continue
        if migrate_file(f):
            count += 1

    print(f"Migration complete. Modified {count} files.")

if __name__ == "__main__":
    main()
