"""Test script to verify initial deck placement works correctly."""
import os
import sys

# Ensure Python fallback is used
os.environ['DM_DISABLE_NATIVE'] = '1'
os.environ['PYTHONPATH'] = r'C:\Users\ichirou\DM_simulation'

sys.path.insert(0, r'C:\Users\ichirou\DM_simulation')

import dm_ai_module as dm

print("Testing initial deck placement...")
print()

# Create game instance
seed = 42
try:
    # Try to load card database
    card_db = dm.JsonLoader.load_cards("data/cards.json")
    print("✓ Loaded card database")
except Exception as e:
    print(f"✗ Failed to load card database: {e}")
    card_db = None

# Create GameInstance
game = dm.GameInstance(seed, card_db)
gs = game.state

# Setup test duel
gs.setup_test_duel()
print("✓ Setup test duel")

# Create deck lists (40 cards each)
deck0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4
deck1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4

# Set decks
gs.set_deck(0, deck0)
gs.set_deck(1, deck1)
print(f"✓ Set decks - P0: {len(gs.players[0].deck)} cards, P1: {len(gs.players[1].deck)} cards")

# Check deck cards are CardStub objects
if gs.players[0].deck:
    print(f"✓ Deck cards are {type(gs.players[0].deck[0]).__name__} objects")

# Start game (should place 5 shields and draw 5 cards for each player)
dm.PhaseManager.start_game(gs, card_db)
print("✓ Called PhaseManager.start_game()")

print()
print("=== After start_game ===")
print(f"Player 0:")
print(f"  Deck: {len(gs.players[0].deck)} cards")
print(f"  Hand: {len(gs.players[0].hand)} cards")
print(f"  Shields: {len(gs.players[0].shield_zone)} cards")
print(f"  Mana: {len(gs.players[0].mana_zone)} cards")
print(f"  Battle: {len(gs.players[0].battle_zone)} cards")

print()
print(f"Player 1:")
print(f"  Deck: {len(gs.players[1].deck)} cards")
print(f"  Hand: {len(gs.players[1].hand)} cards")
print(f"  Shields: {len(gs.players[1].shield_zone)} cards")
print(f"  Mana: {len(gs.players[1].mana_zone)} cards")
print(f"  Battle: {len(gs.players[1].battle_zone)} cards")

print()
# Verify expected state
expected_deck = 30  # 40 - 5 shields - 5 hand
expected_hand = 5
expected_shields = 5

all_good = True
if len(gs.players[0].deck) != expected_deck:
    print(f"✗ P0 deck size mismatch: expected {expected_deck}, got {len(gs.players[0].deck)}")
    all_good = False
if len(gs.players[0].hand) != expected_hand:
    print(f"✗ P0 hand size mismatch: expected {expected_hand}, got {len(gs.players[0].hand)}")
    all_good = False
if len(gs.players[0].shield_zone) != expected_shields:
    print(f"✗ P0 shield count mismatch: expected {expected_shields}, got {len(gs.players[0].shield_zone)}")
    all_good = False

if len(gs.players[1].deck) != expected_deck:
    print(f"✗ P1 deck size mismatch: expected {expected_deck}, got {len(gs.players[1].deck)}")
    all_good = False
if len(gs.players[1].hand) != expected_hand:
    print(f"✗ P1 hand size mismatch: expected {expected_hand}, got {len(gs.players[1].hand)}")
    all_good = False
if len(gs.players[1].shield_zone) != expected_shields:
    print(f"✗ P1 shield count mismatch: expected {expected_shields}, got {len(gs.players[1].shield_zone)}")
    all_good = False

if all_good:
    print("✅ All deck placement checks passed!")
else:
    print("❌ Some checks failed")
    sys.exit(1)
