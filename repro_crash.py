
import sys
import os

try:
    import dm_ai_module
    print("Successfully imported dm_ai_module")
except ImportError as e:
    print(f"Failed to import dm_ai_module: {e}")
    sys.exit(1)

def repro():
    print("Attempting to load cards via JsonLoader...")
    try:
        # Load the cards - this returns a shared_ptr<CardDatabase> (map<int, CardDefinition>)
        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        print(f"CardDB loaded. Type: {type(card_db)}")
        
        # Taking the length (bind_map support)
        print(f"CardDB size: {len(card_db)}")
        
        # Access a card to trigger casting
        # If CardDefinition is bound with shared_ptr holder, but map contains values, this should crash
        print("Accessing a card (id=1)...")
        if 4006 in card_db: # Use a known ID, e.g. 4006 (Bronze-Arm Tribe) or just 1 if exists.
                             # data/cards.json usually has proper IDs.
            card = card_db[4006]
            print(f"Got card: {card.name}")
        else:
             # Fallback to iteration to find any card
             print("Key 4006 not found, iterating keys...")
             for k in card_db:
                 print(f"Accessing key {k}...")
                 card = card_db[k]
                 print(f"Got card: {card.name}")
                 break
                 
        print("SUCCESS: No crash occurred.")
        
    except Exception as e:
        print(f"Caught exception: {e}")

if __name__ == "__main__":
    repro()
