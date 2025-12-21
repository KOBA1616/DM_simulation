import sys
import os
import csv

# Add python/ and root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import dm_ai_module

def test_card_loading():
    print("Testing Card Loading from CSV...")

    # Create a temporary CSV file
    csv_path = "temp_test_cards.csv"

    # Header + 1 Card
    # ID,Name,Civilization,Type,Cost,Power,Races,Keywords
    # Note: Races and Keywords are semicolon separated
    header = ["ID", "Name", "Civilization", "Type", "Cost", "Power", "Races", "Keywords"]
    card_data = ["9999", "Test Dragon", "FIRE", "CREATURE", "5", "5000", "Armored Dragon;Fire Bird", "SPEED_ATTACKER;POWER_ATTACKER"]

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(card_data)

        print(f"Created temporary CSV: {csv_path}")

        # Load using C++ Loader
        card_db = dm_ai_module.CsvLoader.load_cards(csv_path)

        if 9999 in card_db:
            card = card_db[9999]
            print(f"Successfully loaded card ID {card.id}")
            print(f"Name: {card.name}")
            print(f"Civilization: {card.civilization}") # Enum value
            print(f"Type: {card.type}") # Enum value
            print(f"Cost: {card.cost}")
            print(f"Power: {card.power}")
            print(f"Races: {card.races}")

            # Check Keywords
            # Keywords are boolean flags in the struct, exposed as properties or a dict?
            # Let's check the bindings. usually exposed as properties of `keywords` struct member.
            # But in Python binding, `card.keywords` might be an object.

            print(f"Keywords object: {card.keywords}")
            print(f"Speed Attacker: {card.keywords.speed_attacker}")
            print(f"Power Attacker: {card.keywords.power_attacker}")
            print(f"Blocker: {card.keywords.blocker}")

            # Validation
            assert card.name == "Test Dragon"
            assert card.cost == 5
            assert card.power == 5000
            assert "Armored Dragon" in card.races
            assert "Fire Bird" in card.races
            assert card.keywords.speed_attacker == True
            assert card.keywords.power_attacker == True
            assert card.keywords.blocker == False

            print("Validation Passed!")

        else:
            print("Error: Card ID 9999 not found in loaded DB.")

    except Exception as e:
        print(f"Test Failed with Exception: {e}")
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print("Removed temporary CSV.")

if __name__ == "__main__":
    test_card_loading()
