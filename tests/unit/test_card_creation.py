import sys
import os
import json

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dm_ai_module

def test_card_loading():
    print("Testing Card Loading from JSON...")

    # Create a temporary JSON file
    json_path = "temp_test_cards.json"

    # Card Data
    card_data = {
        "id": 9999,
        "name": "Test Dragon",
        "civilizations": ["FIRE"],
        "type": "CREATURE",
        "cost": 5,
        "power": 5000,
        "races": ["Armored Dragon", "Fire Bird"],
        "keywords": {
            "speed_attacker": True
        }
    }

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([card_data], f)

        print(f"Created temporary JSON: {json_path}")

        # Load using C++ Loader
        card_db = dm_ai_module.JsonLoader.load_cards(json_path)

        if 9999 in card_db:
            card = card_db[9999]
            print(f"Successfully loaded card ID {card.id}")
            print(f"Name: {card.name}")
            print(f"Civilization: {card.civilization}") # Enum value
            print(f"Type: {card.type}") # Enum value
            print(f"Cost: {card.cost}")
            print(f"Power: {card.power}")
            print(f"Races: {card.races}")

            print(f"Keywords object: {card.keywords}")
            print(f"Speed Attacker: {card.keywords.speed_attacker}")
            print(f"Blocker: {card.keywords.blocker}")

            # Validation
            assert card.name == "Test Dragon"
            assert card.cost == 5
            assert card.power == 5000
            assert "Armored Dragon" in card.races
            assert "Fire Bird" in card.races
            assert card.keywords.speed_attacker == True
            assert card.keywords.blocker == False

            print("Validation Passed!")

        else:
            print("Error: Card ID 9999 not found in loaded DB.")

    except Exception as e:
        print(f"Test Failed with Exception: {e}")
    finally:
        # Cleanup
        if os.path.exists(json_path):
            os.remove(json_path)
            print("Removed temporary JSON.")

if __name__ == "__main__":
    test_card_loading()
