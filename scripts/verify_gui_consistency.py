import json
import os
import sys

# Add repository root to path
sys.path.append(os.getcwd())

from dm_toolkit.consts import EDITOR_ACTION_TYPES
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

def check_consistency():
    print("Loading command_ui.json...")
    with open("data/configs/command_ui.json", "r", encoding="utf-8") as f:
        command_ui = json.load(f)

    ui_keys = set(command_ui.keys())
    if "COMMAND_GROUPS" in ui_keys:
        ui_keys.remove("COMMAND_GROUPS")

    # 1. Check EDITOR_ACTION_TYPES coverage
    print("\n--- Checking EDITOR_ACTION_TYPES coverage ---")
    editor_keys = set(EDITOR_ACTION_TYPES)
    missing_in_editor = ui_keys - editor_keys
    if missing_in_editor:
        print(f"MISSING in EDITOR_ACTION_TYPES ({len(missing_in_editor)}):")
        for k in sorted(missing_in_editor):
            print(f"  - {k}")
    else:
        print("EDITOR_ACTION_TYPES covers all UI keys.")

    # 2. Check ACTION_MAP coverage
    print("\n--- Checking ACTION_MAP coverage in TextResources ---")
    action_map_keys = set(CardTextResources.ACTION_MAP.keys())

    # Some keys might be handled dynamically or mapped (e.g. MANA_CHARGE -> ADD_MANA)
    # But strictly speaking, if it's a type in the UI, we should probably have a template or logic for it.
    missing_in_action_map = ui_keys - action_map_keys

    # Filter out keys that might be handled specially by TextGenerator
    # TextGenerator often handles things via if/elif blocks before checking ACTION_MAP.
    # We will verify this by running the generator next.
    if missing_in_action_map:
        print(f"MISSING in ACTION_MAP ({len(missing_in_action_map)}):")
        for k in sorted(missing_in_action_map):
            print(f"  - {k}")
    else:
        print("ACTION_MAP covers all UI keys.")

    # 3. Verify Text Generation for each type
    print("\n--- Verifying Text Generation ---")
    failed_types = []
    for key in sorted(ui_keys):
        # Create a dummy action
        dummy_action = {"type": key, "target_group": "OPPONENT", "amount": 1}

        # Some types require specific fields to avoid crashing or empty output
        if key == "IF":
            dummy_action["condition"] = {"type": "SHIELD_COUNT", "value": 0}
            dummy_action["if_true"] = [{"type": "DRAW_CARD", "amount": 1}]
        elif key == "IF_ELSE":
            dummy_action["condition"] = {"type": "SHIELD_COUNT", "value": 0}
            dummy_action["if_true"] = [{"type": "DRAW_CARD", "amount": 1}]
            dummy_action["if_false"] = [{"type": "DISCARD", "amount": 1}]
        elif key == "SELECT_OPTION":
            dummy_action["options"] = [[{"type": "DRAW_CARD"}], [{"type": "DISCARD"}]]
        elif key == "PLAY_FROM_ZONE":
            dummy_action["from_zone"] = "GRAVEYARD"
        elif key == "MEKRAID":
             dummy_action["value1"] = 3
             dummy_action["value2"] = 3
        elif key == "FRIEND_BURST":
             dummy_action["str_val"] = "Race"

        try:
            # We wrap it in a dummy card structure because generator expects data dict
            # Actually generate_text expects full card data, but _format_command is what we care about.
            # But we can call _format_command directly.
            text = CardTextGenerator._format_command(dummy_action)
            if not text:
                print(f"WARNING: No text generated for {key}")
            # else:
            #    print(f"OK: {key} -> {text}")
        except Exception as e:
            print(f"ERROR: Generation failed for {key}: {e}")
            failed_types.append(key)

    if failed_types:
        print(f"\nFailed Types: {failed_types}")
    else:
        print("\nAll types generated text successfully (no crashes).")

if __name__ == "__main__":
    check_consistency()
