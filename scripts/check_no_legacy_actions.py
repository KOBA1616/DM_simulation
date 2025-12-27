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

    def check_node(node, path="root"):
        local_fails = 0
        if isinstance(node, dict):
            if 'actions' in node:
                # Ignore empty actions list if allowed? No, strict migration implies removal of key.
                print(f"FAILURE: 'actions' field found in {filepath} at {path}")
                local_fails += 1

            # Recurse
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    local_fails += check_node(v, f"{path}.{k}")

        elif isinstance(node, list):
            for i, item in enumerate(node):
                local_fails += check_node(item, f"{path}[{i}]")

        return local_fails

    fail_count += check_node(data)
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
