import json
import os
import sys

# Add repo root to path
sys.path.append(os.getcwd())

from dm_toolkit.gui.editor.schema_def import SchemaLoader, FieldType
from dm_toolkit.gui.editor.text_resources import CardTextResources

def verify():
    # Load command_ui.json
    config_path = os.path.join('data', 'configs', 'command_ui.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        command_ui = json.load(f)

    # Load ACTION_MAP
    action_map = CardTextResources.ACTION_MAP

    print("--- Verifying Command UI vs Text Resources ---")

    # Check 1: Commands in ACTION_MAP but not in command_ui
    missing_in_ui = []
    for cmd in action_map.keys():
        if cmd not in command_ui:
            # Filter internal/generic mappings that might not be commands
            if cmd in ["TRANSITION", "MUTATE", "IF", "IF_ELSE", "ELSE"]: continue
            missing_in_ui.append(cmd)

    if missing_in_ui:
        print(f"Commands in ACTION_MAP but missing in command_ui.json: {missing_in_ui}")

    # Check 2: Fields missing from _DEFAULT_MAPPING in schema_def.py
    # We instantiate SchemaLoader to check _DEFAULT_MAPPING
    mapping = SchemaLoader._DEFAULT_MAPPING

    needed_keys = ['result', 'query_mode', 'option_count']
    missing_mappings = [k for k in needed_keys if k not in mapping]

    if missing_mappings:
        print(f"Keys missing from SchemaLoader._DEFAULT_MAPPING: {missing_mappings}")

    # Check 3: FLOW command fields
    if "FLOW" in command_ui:
        visible = command_ui["FLOW"].get("visible", [])
        if "amount" not in visible:
            print("FLOW command missing 'amount' field (needed for PHASE_CHANGE val1).")

    # Check 4: ATTACH command
    if "ATTACH" not in command_ui:
        print("ATTACH command missing from command_ui.json")

    # Check 5: COST_REDUCTION
    if "COST_REDUCTION" not in command_ui:
        print("COST_REDUCTION command missing from command_ui.json")

if __name__ == "__main__":
    verify()
