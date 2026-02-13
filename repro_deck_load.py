
import json
import os
import sys

# Mocking parts of the environment
class MockDeckBuilder:
    def __init__(self):
        self.card_db = {1: {'id': 1, 'name': 'Card1'}, 2: {'id': 2, 'name': 'Card2'}}
        self.current_deck = []

    def load_deck(self, fname):
        print(f"Loading deck from {fname}")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    self.current_deck = json.load(f)
                print(f"Loaded deck content type: {type(self.current_deck)}")
                print(f"Loaded deck content length: {len(self.current_deck)}")
                self.update_deck_list()
            except Exception as e:
                print(f"Failed to load deck: {e}")
                import traceback
                traceback.print_exc()

    def update_deck_list(self):
        print("Updating deck list...")
        self.current_deck.sort()
        for cid in self.current_deck:
            if cid in self.card_db:
                # print(f"Card {cid} found in DB")
                pass
            else:
                # print(f"Card {cid} NOT found in DB")
                pass
        print("Deck list updated.")

if __name__ == "__main__":
    builder = MockDeckBuilder()
    # Try loading the existing Japanese filename deck
    # Note: encoding might be an issue if the system locale is not utf-8 compatible for filenames, 
    # but Python 3 usually handles it well.
    deck_path = "data/decks/緑単3コス.json"
    
    if os.path.exists(deck_path):
        print(f"File {deck_path} exists.")
        builder.load_deck(deck_path)
    else:
        print(f"File {deck_path} does NOT exist.")
        # Try listing dir to see actual name
        print("Listing data/decks:")
        for f in os.listdir("data/decks"):
            print(f" - {f}")
            if "緑単" in f:
                print(f"   (Matched '緑単': {f})")
                builder.load_deck(os.path.join("data/decks", f))

