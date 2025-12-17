import json
import os
import sys

def convert_legacy_action(action):
    cmd = {
        "optional": action.get("optional", False),
        "amount": action.get("value1", 0),
        "str_param": action.get("str_val", ""),
        "target_filter": action.get("filter", {}),
        "target_group": action.get("scope", "NONE"),
        "from_zone": action.get("source_zone", ""),
        "to_zone": action.get("destination_zone", "")
    }

    action_type = action.get("type", "NONE")

    # Check for variable linking (unsupported in CommandDef currently)
    if "input_value_key" in action and action["input_value_key"]:
        return None
    if "output_value_key" in action and action["output_value_key"]:
        return None

    if action_type == "DRAW_CARD":
        cmd["type"] = "DRAW_CARD"
    elif action_type == "ADD_MANA":
        cmd["type"] = "MANA_CHARGE"
    elif action_type == "DESTROY":
        cmd["type"] = "DESTROY"
    elif action_type == "RETURN_TO_HAND":
        cmd["type"] = "RETURN_TO_HAND"
    elif action_type == "TAP":
        cmd["type"] = "TAP"
    elif action_type == "UNTAP":
        cmd["type"] = "UNTAP"
    elif action_type == "MODIFY_POWER":
        cmd["type"] = "POWER_MOD"
    elif action_type == "BREAK_SHIELD":
        cmd["type"] = "BREAK_SHIELD"
    elif action_type == "DISCARD":
        cmd["type"] = "DISCARD"
        # Fix: Discard goes to Graveyard
        if not cmd["to_zone"]:
             cmd["to_zone"] = "GRAVEYARD"
    elif action_type == "SEARCH_DECK":
        cmd["type"] = "SEARCH_DECK"
    elif action_type == "GRANT_KEYWORD":
        cmd["type"] = "ADD_KEYWORD"
    elif action_type == "SEND_TO_MANA":
        cmd["type"] = "MANA_CHARGE"
    elif action_type == "MOVE_CARD":
        cmd["type"] = "TRANSITION"
    else:
        # Not convertible or handled by legacy fallback
        return None

    # Clean up empty fields to minimize JSON size
    if not cmd["str_param"]: del cmd["str_param"]
    if not cmd["target_filter"]: del cmd["target_filter"]
    if cmd["target_group"] == "NONE": del cmd["target_group"]
    if not cmd["from_zone"]: del cmd["from_zone"]
    if not cmd["to_zone"]: del cmd["to_zone"]
    if cmd["amount"] == 0: del cmd["amount"]

    return cmd

def migrate_cards(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    migrated_count = 0

    def process_effects(effects, card_id):
        nonlocal migrated_count
        for effect in effects:
            if "actions" in effect and effect["actions"] and ("commands" not in effect or not effect["commands"]):
                new_commands = []
                all_convertible = True

                for action in effect["actions"]:
                    cmd = convert_legacy_action(action)
                    if cmd:
                        new_commands.append(cmd)
                    else:
                        all_convertible = False
                        action_type = action.get("type", "UNKNOWN")
                        # print(f"Card {card_id}: Skipping action {action_type} - not convertible.")

                if new_commands and all_convertible:
                     effect["commands"] = new_commands
                     del effect["actions"]
                     migrated_count += 1
                # elif new_commands and not all_convertible:
                    # print(f"Card {card_id}: Partial migration not supported. Keeping legacy actions.")

    for card in cards:
        if "effects" in card:
            process_effects(card["effects"], card["id"])

        if "spell_side" in card and card["spell_side"]:
             if "effects" in card["spell_side"]:
                 process_effects(card["spell_side"]["effects"], card["id"])

    with open(file_path, 'w', encoding='utf-8') as f:
        # Use indent=2 for consistent formatting
        json.dump(cards, f, indent=2, ensure_ascii=False)

    print(f"Migrated {migrated_count} effects.")

if __name__ == "__main__":
    migrate_cards("data/cards.json")
