import json
import pytest
import os
from dm_ai_module import JsonLoader, CardDefinition, CardKeywords, FilterDef, Civilization

def test_load_revolution_change_json():
    # 1. Create a temporary JSON file
    json_content = """
    [
        {
            "id": 800,
            "name": "Revolution Dragon",
            "civilization": "FIRE",
            "type": "CREATURE",
            "cost": 7,
            "power": 7000,
            "races": ["Mega Command Dragon"],
            "effects": [],
            "revolution_change_condition": {
                "civilizations": ["FIRE"],
                "races": ["Dragon"],
                "min_cost": 5
            }
        }
    ]
    """

    filename = "temp_revolution_test.json"
    try:
        with open(filename, "w") as f:
            f.write(json_content)

        # 2. Load the JSON
        card_db = JsonLoader.load_cards(filename)

        # 3. Verify
        assert 800 in card_db
        card = card_db[800]

        # Check basic properties
        assert card.name == "Revolution Dragon"
        assert card.civilization == Civilization.FIRE

        # Check Keyword Flag
        assert card.keywords.revolution_change == True

        # Check Condition (using python bindings if exposed, otherwise implicit check via optional presence)
        # The binding for CardDefinition exposes revolution_change_condition as Optional[FilterDef]
        cond = card.revolution_change_condition
        assert cond is not None
        assert cond.civilizations == ["FIRE"]
        assert cond.races == ["Dragon"]
        assert cond.min_cost == 5

        print("JSON Loading for Revolution Change Verified!")

    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_load_revolution_change_json()
