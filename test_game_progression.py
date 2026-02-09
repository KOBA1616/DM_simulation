"""Test script to verify game progression works correctly."""
import os
import sys

os.environ['DM_DISABLE_NATIVE'] = '1'
os.environ['PYTHONPATH'] = r'C:\Users\ichirou\DM_simulation'
sys.path.insert(0, r'C:\Users\ichirou\DM_simulation')

import dm_ai_module as dm

print("Testing game progression...")
print()

# Load card database
try:
    card_db = dm.JsonLoader.load_cards("data/cards.json")
    print("✓ Loaded card database")
except Exception as e:
    print(f"✗ Failed to load card database: {e}")
    sys.exit(1)

# Create game instance
game = dm.GameInstance(42, card_db)
gs = game.state

# Setup
gs.setup_test_duel()
deck0 = [1, 2, 3, 4, 5] * 8
deck1 = [1, 2, 3, 4, 5] * 8
gs.set_deck(0, deck0)
gs.set_deck(1, deck1)

print(f"✓ Setup complete")
print(f"  P0 deck: {len(gs.players[0].deck)} cards")
print(f"  P1 deck: {len(gs.players[1].deck)} cards")
print()

# Start game
dm.PhaseManager.start_game(gs, card_db)
print(f"✓ Game started")
print(f"  Current phase: {gs.current_phase}")
print(f"  Active player: {gs.active_player_id}")
print(f"  P0 hand: {len(gs.players[0].hand)}, shields: {len(gs.players[0].shield_zone)}, deck: {len(gs.players[0].deck)}")
print(f"  P1 hand: {len(gs.players[1].hand)}, shields: {len(gs.players[1].shield_zone)}, deck: {len(gs.players[1].deck)}")
print()

# Try to generate actions
print("Generating legal actions...")
try:
    from dm_toolkit import commands_v2
    actions = commands_v2.generate_legal_commands(gs, card_db)
    print(f"✓ Generated {len(actions)} legal actions")
    if actions:
        for i, action in enumerate(actions[:5]):
            print(f"  Action {i}: {action}")
except Exception as e:
    print(f"✗ Failed to generate actions: {e}")
    import traceback
    traceback.print_exc()

print()

# Try game step
print("Testing game.step()...")
try:
    result = game.step()
    print(f"✓ game.step() returned: {result}")
    print(f"  Current phase: {gs.current_phase}")
    print(f"  Active player: {gs.active_player_id}")
    print(f"  Turn number: {gs.turn_number}")
except Exception as e:
    print(f"✗ game.step() failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Try fast_forward
print("Testing PhaseManager.fast_forward()...")
try:
    dm.PhaseManager.fast_forward(gs, card_db)
    print(f"✓ fast_forward() completed")
    print(f"  Current phase: {gs.current_phase}")
    print(f"  Active player: {gs.active_player_id}")
except Exception as e:
    print(f"✗ fast_forward() failed: {e}")
    import traceback
    traceback.print_exc()
