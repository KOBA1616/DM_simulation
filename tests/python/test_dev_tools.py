import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import dm_ai_module

def test_dev_tools():
    print("Testing DevTools...")
    try:
        gs = dm_ai_module.GameState(123)
        gs.setup_test_duel()
        
        # Check initial deck size
        print(f"Initial Deck Size P0: {len(gs.players[0].deck)}")
        print(f"Initial Hand Size P0: {len(gs.players[0].hand)}")
        
        # Move 5 cards from Deck to Hand
        count = dm_ai_module.DevTools.move_cards(
            gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.HAND, 5
        )
        print(f"Moved {count} cards from Deck to Hand.")
        
        print(f"New Deck Size P0: {len(gs.players[0].deck)}")
        print(f"New Hand Size P0: {len(gs.players[0].hand)}")
        
        if len(gs.players[0].hand) == 5 and len(gs.players[0].deck) == 35:
            print("SUCCESS: Cards moved correctly.")
        else:
            print("FAILURE: Card counts incorrect.")
            
        # Test Filter: Move specific card ID (1) from Hand to Mana
        # Assuming ID 1 exists in hand (setup_test_duel puts ID 1 in deck, we just moved them to hand)
        count = dm_ai_module.DevTools.move_cards(
            gs, 0, dm_ai_module.Zone.HAND, dm_ai_module.Zone.MANA, 1, 1
        )
        print(f"Moved {count} cards with ID 1 from Hand to Mana.")
        
        print(f"New Hand Size P0: {len(gs.players[0].hand)}")
        print(f"New Mana Size P0: {len(gs.players[0].mana_zone)}")
        
        if count == 1 and len(gs.players[0].mana_zone) == 1:
             print("SUCCESS: Filtered move worked.")
        else:
             print("FAILURE: Filtered move failed.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dev_tools()
