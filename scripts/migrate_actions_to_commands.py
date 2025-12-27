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

    def process_node(node):
        nonlocal modified
        if isinstance(node, dict):
            # 1. Migrate actions -> commands
            if 'actions' in node:
                actions = node.get('actions')
                if isinstance(actions, list):
                    commands = node.get('commands', [])
                    for act in actions:
                        if isinstance(act, dict):
                            cmd = ActionToCommandMapper.map_action(act)
                            commands.append(cmd)
                    node['commands'] = commands
                    del node['actions']
                    modified = True

            # 2. Recurse into children
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    process_node(value)

        elif isinstance(node, list):
            for item in node:
                process_node(item)

    process_node(data)

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
