
import dm_ai_module
import os
import sys

# Ensure current directory and dm_toolkit are in path for module import
sys.path.append(os.getcwd())

def check_initialization():
    print("Checking Game Initialization...")
    try:
        from dm_ai_module import GameInstance, JsonLoader
    except ImportError:
        print("Error: Could not import dm_ai_module.")
        return

    game = GameInstance()
    # Load cards (needed for proper game start if cards are used)
    # card_db = JsonLoader.load_cards("data/cards.json") 
    
    # Set default decks (required for start_game to work)
    default_deck = [1] * 40
    game.state.set_deck(0, default_deck)
    game.state.set_deck(1, default_deck)

    # Initialize game
    print("Calling game.start_game()...")
    game.start_game()
    
    # Check integrity
    p0 = game.state.players[0]
    p1 = game.state.players[1]
    
    print(f"Player 0 Hand: {len(p0.hand)} (Expected 5)")
    print(f"Player 0 Shields: {len(p0.shield_zone)} (Expected 5)")
    print(f"Player 1 Hand: {len(p1.hand)} (Expected 5)")
    print(f"Player 1 Shields: {len(p1.shield_zone)} (Expected 5)")
    
    if len(p0.hand) == 5 and len(p0.shield_zone) == 5:
        print("SUCCESS: Initialization Integrity verified.")
    else:
        print("FAILURE: Initialization Integrity check failed.")

if __name__ == "__main__":
    check_initialization()
