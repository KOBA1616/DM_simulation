import json
import os

def repair_and_format():
    file_path = "data/cards.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    for card in cards:
        # Repair ID 11 (Napoleon Vibes) - Revert to legacy actions
        if card["id"] == 11:
            if "effects" in card and len(card["effects"]) > 0:
                eff = card["effects"][0]
                # If it has commands and no actions (or we just want to force overwrite)
                # We restore actions
                eff["actions"] = [
                    {
                        "destination_zone": "HAND",
                        "filter": {},
                        "input_value_key": "",
                        "optional": True,
                        "output_value_key": "var_discarded_count",
                        "scope": "PLAYER_SELF",
                        "str_val": "",
                        "type": "DISCARD",
                        "value1": 2,
                        "value2": 0
                    },
                    {
                        "destination_zone": "HAND",
                        "filter": {},
                        "input_value_key": "var_discarded_count",
                        "optional": False,
                        "output_value_key": "",
                        "scope": "PLAYER_SELF",
                        "str_val": "",
                        "type": "DRAW_CARD",
                        "value1": 0,
                        "value2": 0
                    },
                    {
                        "destination_zone": "HAND",
                        "filter": {},
                        "input_value_key": "",
                        "optional": False,
                        "output_value_key": "",
                        "scope": "PLAYER_SELF",
                        "str_val": "",
                        "type": "DRAW_CARD",
                        "value1": 1,
                        "value2": 0
                    }
                ]
                if "commands" in eff:
                    del eff["commands"]
                print("Repaired Card ID 11.")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)
    print("Formatted data/cards.json with indent=2.")

if __name__ == "__main__":
    repair_and_format()
