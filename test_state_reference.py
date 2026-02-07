"""Test if game_instance.state returns reference or copy."""
import sys
sys.path.insert(0, '.')
import dm_ai_module

seed = 42
card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
game_instance = dm_ai_module.GameInstance(seed, card_db)

# Get state multiple times
gs1 = game_instance.state
gs2 = game_instance.state
gs3 = game_instance.state

# Check if they are the same object
print(f"gs1 is gs2: {gs1 is gs2}")
print(f"gs2 is gs3: {gs2 is gs3}")
print(f"gs1 is gs3: {gs1 is gs3}")

print(f"\nid(gs1): {id(gs1)}")
print(f"id(gs2): {id(gs2)}")
print(f"id(gs3): {id(gs3)}")

# Test if modifications persist
gs1.setup_test_duel()
deck = [1,2,3,4,5,6,7,8,9,10]*4
gs1.set_deck(0, deck)
gs1.set_deck(1, deck)

dm_ai_module.PhaseManager.start_game(gs1, card_db)
print(f"\nAfter modifying gs1:")
print(f"  gs1.turn_number: {gs1.turn_number}")

# Get new reference
gs_new = game_instance.state
print(f"  gs_new.turn_number: {gs_new.turn_number}")
print(f"  gs_new is gs1: {gs_new is gs1}")

# Modify gs1 and check if gs_new sees it
gs1.turn_number = 99
print(f"\nAfter setting gs1.turn_number = 99:")
print(f"  gs1.turn_number: {gs1.turn_number}")
print(f"  gs_new.turn_number: {gs_new.turn_number}")
