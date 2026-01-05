
import sys
import os

# Add bin/ to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module")
    sys.exit(1)

def test_memory_management():
    # Initialize GameState
    game = dm_ai_module.GameState(100) # 100 cards

    # Check initial players
    players_copy_1 = game.players
    print(f"Len players_copy_1: {len(players_copy_1)}")

    # Verify initially empty
    print(f"Copy 1 hand size (initial): {len(players_copy_1[0].hand)}")

    # Add a card to hand via API
    print("Adding card via add_card_to_hand...")
    game.add_card_to_hand(0, 1, 100) # pid=0, cid=1, iid=100

    # Check copy 1 again
    print(f"Copy 1 hand size (after add): {len(players_copy_1[0].hand)}")

    # Fetch new copy
    players_copy_2 = game.players
    print(f"Copy 2 hand size (new fetch): {len(players_copy_2[0].hand)}")

    if len(players_copy_1[0].hand) == 0 and len(players_copy_2[0].hand) == 1:
        print("CONFIRMED: game.players returns a COPY.")
    elif len(players_copy_1[0].hand) == 1:
        print("OBSERVATION: game.players seems to return a REFERENCE or proxy.")
    else:
        print("UNKNOWN STATE")

    # Now test direct list modification
    print("Appending to Copy 2 hand...")
    card = dm_ai_module.CardInstance()
    card.instance_id = 101
    card.card_id = 1
    card.owner = 0
    players_copy_2[0].hand.append(card)

    print(f"Copy 2 hand size (after append): {len(players_copy_2[0].hand)}")

    # Fetch Copy 3
    players_copy_3 = game.players
    print(f"Copy 3 hand size (new fetch): {len(players_copy_3[0].hand)}")

    if len(players_copy_3[0].hand) == 1:
        print("CONFIRMED: Direct list modification did NOT persist.")
    elif len(players_copy_3[0].hand) == 2:
        print("SUCCESS: Direct list modification persisted!")

if __name__ == "__main__":
    test_memory_management()
