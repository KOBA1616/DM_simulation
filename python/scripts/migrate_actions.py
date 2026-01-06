import os
import json
import glob
import sys
import copy

# Add repository root to path to import dm_toolkit
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from dm_toolkit.action_to_command import action_to_command
except ImportError:
    print("Error: Could not import dm_toolkit.action_to_command")
    sys.exit(1)

def migrate_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return

    modified = False

    if isinstance(data, list):
        # List of cards
        for card in data:
            if traverse_and_migrate(card):
                modified = True
    elif isinstance(data, dict):
        # Single card or object
        if traverse_and_migrate(data):
            modified = True

    if modified:
        print(f"Migrated {filepath}")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save {filepath}: {e}")
    else:
        print(f"No changes for {filepath}")

def traverse_and_migrate(node):
    modified = False
    if isinstance(node, dict):
        # Check for 'actions' list
        if 'actions' in node and isinstance(node['actions'], list):
            # If commands don't exist, create them
            if 'commands' not in node:
                node['commands'] = []

            # Append converted actions to commands
            for action in node['actions']:
                cmd = action_to_command(action)
                node['commands'].append(cmd)

            # Remove actions
            del node['actions']
            modified = True

        # Recursive traversal
        for k, v in node.items():
            if traverse_and_migrate(v):
                modified = True

    elif isinstance(node, list):
        for item in node:
            if traverse_and_migrate(item):
                modified = True

    return modified

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    # Find all json files
    files = glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True)

    print(f"Scanning {len(files)} files in {data_dir}...")
    for f in files:
        migrate_file(f)
