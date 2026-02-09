"""Test GameInstance directly without PhaseManager."""
import sys
sys.path.insert(0, '.')
import dm_ai_module
from dm_toolkit import commands_v2 as commands

# Create game
seed = 42
card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
game_instance = dm_ai_module.GameInstance(seed, card_db)
gs = game_instance.state
gs.setup_test_duel()

deck = [1,2,3,4,5,6,7,8,9,10]*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)

# Start game
dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)  # Need initial fast_forward
gs = game_instance.state  # Re-sync after fast_forward

print(f"After start_game + fast_forward: phase={gs.current_phase}, turn={gs.turn_number}")

# Generate actions
actions = commands.generate_legal_commands(gs, card_db, strict=False)
print(f"Actions: {len(actions)}")

# Find MA NA_CHARGE
mana_action = None
for a in actions:
    if int(a.type) == 1:  # MANA_CHARGE
        mana_action = a
        break

if mana_action:
    print(f"\nExecuting MANA_CHARGE card_id={mana_action.card_id}")
    print(f"Before: phase={gs.current_phase}, mana={len(gs.players[0].mana_zone)}")
    
    # Execute via GameInstance
    game_instance.resolve_action(mana_action)
    
    # Re-sync gs
    gs = game_instance.state
    
    print(f"After resolve_action: phase={gs.current_phase}, mana={len(gs.players[0].mana_zone)}")
    
    # Now call fast_forward
    dm_ai_module.PhaseManager.fast_forward(gs, card_db)
    
    # Re-sync again
    gs = game_instance.state
    
    print(f"After fast_forward: phase={gs.current_phase}, mana={len(gs.players[0].mana_zone)}")
    
    # Check if we're in MAIN_PHASE
    if gs.current_phase == dm_ai_module.Phase.MAIN:
        print("\n✓ Successfully reached MAIN_PHASE")
        main_actions = commands.generate_legal_commands(gs, card_db, strict=False)
        print(f"MAIN_PHASE actions: {len(main_actions)}")
        for i, a in enumerate(main_actions[:3]):
            print(f"  {i}: type={a.type} card_id={a.card_id}")
    else:
        print(f"\n✗ Still in {gs.current_phase}, not MAIN_PHASE")
