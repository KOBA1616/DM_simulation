"""Extended test for game progression - multiple turns."""
import os
import sys

os.environ['DM_DISABLE_NATIVE'] = '1'
os.environ['PYTHONPATH'] = r'C:\Users\ichirou\DM_simulation'
sys.path.insert(0, r'C:\Users\ichirou\DM_simulation')

import dm_ai_module as dm

print("Testing multi-turn game progression...")
print()

# Load card database
card_db = dm.JsonLoader.load_cards("data/cards.json")

# Create game
game = dm.GameInstance(42, card_db)
gs = game.state

# Setup
gs.setup_test_duel()
gs.set_deck(0, [1, 2, 3, 4, 5] * 8)
gs.set_deck(1, [1, 2, 3, 4, 5] * 8)
dm.PhaseManager.start_game(gs, card_db)

print(f"Initial state:")
print(f"  Phase: {gs.current_phase}, Active: P{gs.active_player_id}, Turn: {gs.turn_number}")
print(f"  P0: hand={len(gs.players[0].hand)}, mana={len(gs.players[0].mana_zone)}, deck={len(gs.players[0].deck)}")
print(f"  P1: hand={len(gs.players[1].hand)}, mana={len(gs.players[1].mana_zone)}, deck={len(gs.players[1].deck)}")
print()

# Run 10 game steps
for step_num in range(1, 11):
    result = game.step()
    if not result:
        print(f"Step {step_num}: Game ended or no actions")
        break
    
    print(f"Step {step_num}: Phase={gs.current_phase.name}, Active=P{gs.active_player_id}, Turn={gs.turn_number}")
    print(f"  P0: hand={len(gs.players[0].hand)}, mana={len(gs.players[0].mana_zone)}, " +
          f"battle={len(gs.players[0].battle_zone)}, deck={len(gs.players[0].deck)}")
    print(f"  P1: hand={len(gs.players[1].hand)}, mana={len(gs.players[1].mana_zone)}, " +
          f"battle={len(gs.players[1].battle_zone)}, deck={len(gs.players[1].deck)}")

print()
print("✅ Multi-turn progression test completed")

# Verify some expected behaviors
print()
print("Verifying expected state:")
if gs.turn_number > 1:
    print(f"✓ Turn counter advanced: {gs.turn_number}")
else:
    print(f"✗ Turn counter didn't advance: {gs.turn_number}")

# At least one player should have mana
total_mana = len(gs.players[0].mana_zone) + len(gs.players[1].mana_zone)
if total_mana > 0:
    print(f"✓ Mana was charged: P0={len(gs.players[0].mana_zone)}, P1={len(gs.players[1].mana_zone)}")
else:
    print(f"✗ No mana charged")

# Deck should be smaller (cards drawn over turns)
initial_deck_size = 30  # 40 - 5 shields - 5 hand
if len(gs.players[0].deck) < initial_deck_size:
    print(f"✓ P0 deck decreased from {initial_deck_size} to {len(gs.players[0].deck)} (cards drawn)")
if len(gs.players[1].deck) < initial_deck_size:
    print(f"✓ P1 deck decreased from {initial_deck_size} to {len(gs.players[1].deck)} (cards drawn)")
