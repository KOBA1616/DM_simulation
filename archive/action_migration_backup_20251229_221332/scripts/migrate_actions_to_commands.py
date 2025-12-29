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
                # SPECIAL CHECK: editor_templates.json uses actions as list of {name, data}
                # Check if elements are wrappers or raw actions
                is_template_wrapper = False
                if isinstance(actions, list) and len(actions) > 0:
                    first = actions[0]
                    if isinstance(first, dict) and 'name' in first and 'data' in first:
                        is_template_wrapper = True

                if is_template_wrapper:
                    # For templates, we migrate the 'data' inside each wrapper
                    # But we also want to rename the root list to 'commands' if we want unified storage?
                    # Or keep 'actions' for legacy templates?
                    # The goal is "Load-Lift". If we change template storage to 'commands',
                    # the editor needs to know to look there.
                    # Current CardDataManager.load_templates loads "commands" and "actions".
                    # Let's migrate the content of 'actions' and move them to 'commands' list in the file.

                    commands = node.get('commands', [])
                    for wrapper in actions:
                        act_data = wrapper.get('data')
                        cmd_data = ActionToCommandMapper.map_action(act_data)
                        commands.append({
                            "name": wrapper.get('name'),
                            "data": cmd_data
                        })
                    node['commands'] = commands
                    del node['actions']
                    modified = True
                else:
                    # Standard migration
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
