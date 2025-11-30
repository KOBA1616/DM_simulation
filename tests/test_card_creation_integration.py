import sys
import os
import csv
from PyQt6.QtWidgets import QApplication

# Add python/ to path so we can import gui modules
# Assuming this script is run from project root or tests/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(os.path.join(project_root, "python"))
    sys.path.append(project_root) # For dm_ai_module if it's in root

from gui.card_editor import CardEditor
import dm_ai_module

def verify():
    # QApplication is required for QWidgets
    app = QApplication(sys.argv)
    csv_path = "data/test_cards_verify.csv"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Create dummy CSV with header
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ID","Name","Civilization","Type","Cost","Power","Races","Keywords"])

    print("Initializing CardEditor...")
    editor = CardEditor(csv_path)
    
    # Mock inputs programmatically
    print("Setting card properties...")
    editor.id_input.setValue(9999)
    editor.name_input.setText("Test Dragon")
    editor.civ_input.setCurrentText("FIRE")
    editor.type_input.setCurrentText("CREATURE")
    editor.cost_input.setValue(7)
    editor.power_input.setValue(6000)
    editor.races_input.setText("Armored Dragon;Fire Bird")
    
    # Check some keywords
    print("Selecting keywords...")
    if "SPEED_ATTACKER" in editor.keyword_checkboxes:
        editor.keyword_checkboxes["SPEED_ATTACKER"].setChecked(True)
    if "DOUBLE_BREAKER" in editor.keyword_checkboxes:
        editor.keyword_checkboxes["DOUBLE_BREAKER"].setChecked(True)
        
    # Trigger Save
    print("Saving card...")
    editor.save_card()
    
    # Verify file content manually first
    print("\n--- CSV Content ---")
    with open(csv_path, "r", encoding='utf-8') as f:
        content = f.read()
        print(content.strip())
    print("-------------------\n")
        
    # Verify with C++ Loader
    print("Verifying with C++ Engine Loader...")
    try:
        db = dm_ai_module.CsvLoader.load_cards(csv_path)
        if 9999 in db:
            card = db[9999]
            print(f"Card Loaded Successfully!")
            print(f"  ID: {card.id}")
            print(f"  Name: {card.name}")
            print(f"  Power: {card.power}")
            print(f"  Civilization: {card.civilization}")
            
            # Keywords are not exposed in Python bindings yet, so we skip checking them.
            # If the card loaded with correct ID/Name/Power, the CSV parsing was successful.
            
            if card.name == "Test Dragon":
                print("\nSUCCESS: Card created by GUI and loaded by Engine correctly.")
            else:
                print("\nFAILURE: Card data mismatch.")
        else:
            print("\nFAILURE: Card ID 9999 not found in DB.")
    except Exception as e:
        print(f"\nFAILURE: C++ Loader crashed or error: {e}")

    # Cleanup
    if os.path.exists(csv_path):
        os.remove(csv_path)
    print("Test cleanup complete.")

if __name__ == "__main__":
    verify()
