"""Debug test to see why mana charge is not happening."""
import os
import sys

os.environ['DM_DISABLE_NATIVE'] = '1'
os.environ['PYTHONPATH'] = r'C:\Users\ichirou\DM_simulation'
sys.path.insert(0, r'C:\Users\ichirou\DM_simulation')

import dm_ai_module as dm
from dm_toolkit import commands_v2

print("Debugging mana charge issue...")
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
print(f"  Phase: {gs.current_phase.name}")
print(f"  Active player: P{gs.active_player_id}")
print(f"  P0 hand: {len(gs.players[0].hand)}, mana: {len(gs.players[0].mana_zone)}")
print()

# Generate actions
print("Generating legal actions...")
actions = commands_v2.generate_legal_commands(gs, card_db)
print(f"Total actions: {len(actions)}")
for i, action in enumerate(actions[:10]):
    action_type = action.get('type') if isinstance(action, dict) else getattr(action, 'type', None)
    print(f"  Action {i}: type={action_type}, instance_id={action.get('source_instance_id') if isinstance(action, dict) else getattr(action, 'source_instance_id', None)}")
print()

# Check which action step() will choose
print("Testing game.step()...")
print(f"Before step - P0 mana: {len(gs.players[0].mana_zone)}, hand: {len(gs.players[0].hand)}")
result = game.step()
print(f"After step - P0 mana: {len(gs.players[0].mana_zone)}, hand: {len(gs.players[0].hand)}")
print(f"Step result: {result}")
print(f"Current phase: {gs.current_phase.name}")
print()

# Try executing a mana charge action directly
print("Manually executing MANA_CHARGE action...")
gs2 = dm.GameState()
gs2.setup_test_duel()
gs2.set_deck(0, [1, 2, 3] * 8)
gs2.set_deck(1, [1, 2, 3] * 8)
dm.PhaseManager.start_game(gs2, card_db)

print(f"Before manual mana charge - P0 hand: {len(gs2.players[0].hand)}, mana: {len(gs2.players[0].mana_zone)}")
if gs2.players[0].hand:
    card = gs2.players[0].hand[0]
    print(f"  Charging card: id={card.card_id}, instance={card.instance_id}")
    
    # Create action
    action = type('Action', (), {})()
    action.type = dm.ActionType.MANA_CHARGE
    action.source_instance_id = card.instance_id
    action.card_id = card.card_id
    
    game2 = dm.GameInstance(0, card_db)
    game2.state = gs2
    game2.execute_action(action)
    
    print(f"After manual mana charge - P0 hand: {len(gs2.players[0].hand)}, mana: {len(gs2.players[0].mana_zone)}")
    if gs2.players[0].mana_zone:
        print(f"  Mana zone contains: {[c.card_id for c in gs2.players[0].mana_zone]}")
