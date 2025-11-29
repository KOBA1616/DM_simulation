import csv
import os
import sys

REQUIRED_COLUMNS = ["ID", "Name", "Civilization", "Type", "Cost", "Power", "Races", "Keywords"]
VALID_CIVILIZATIONS = ["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"]
VALID_TYPES = ["CREATURE", "SPELL", "EVOLUTION_CREATURE", "CROSS_GEAR", "CASTLE", "PSYCHIC_CREATURE", "GR_CREATURE"]

def validate_csv(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check headers
            if not reader.fieldnames:
                print("Error: Empty CSV file")
                return False
                
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing_cols:
                print(f"Error: Missing columns: {missing_cols}")
                return False

            for row_num, row in enumerate(reader, start=2):
                # Validate ID
                try:
                    card_id = int(row["ID"])
                    if card_id <= 0:
                        print(f"Error Line {row_num}: ID must be positive")
                        return False
                except ValueError:
                    print(f"Error Line {row_num}: Invalid ID format")
                    return False

                # Validate Civilization
                civ = row["Civilization"].strip()
                if civ and civ not in VALID_CIVILIZATIONS:
                     print(f"Error Line {row_num}: Invalid Civilization '{civ}'")
                     return False

                # Validate Type
                ctype = row["Type"].strip()
                if ctype and ctype not in VALID_TYPES:
                    print(f"Error Line {row_num}: Invalid Type '{ctype}'")
                    return False

                # Validate Cost/Power
                try:
                    int(row["Cost"])
                    int(row["Power"])
                except ValueError:
                    print(f"Error Line {row_num}: Cost and Power must be integers")
                    return False

        print("Validation Successful: cards.csv is valid.")
        return True

    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), '..', "data", "cards.csv")
    if not validate_csv(csv_path):
        sys.exit(1)
    sys.exit(0)
