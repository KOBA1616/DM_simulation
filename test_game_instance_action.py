"""Test GameInstance.resolve_action() directly"""
import sys
sys.path.insert(0, '.')

import dm_ai_module

# Load card database
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')

# Create GameInstance
gi = dm_ai_module.GameInstance(42, card_db)
state = gi.state
state.setup_test_duel()

# Set decks
deck = [1,2,3,4,5,6,7,8,9,10]*4
state.set_deck(0, deck)
state.set_deck(1, deck)

# Start game
dm_ai_module.PhaseManager.start_game(state, card_db)
print(f"After start_game: phase={state.current_phase}, turn={state.turn_number}")

# Fast forward
dm_ai_module.PhaseManager.fast_forward(state, card_db)
print(f"After fast_forward: phase={state.current_phase}, active_player={state.active_player_id}")

# Get actions
actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
print(f"\nGenerated {len(actions)} actions")

if actions:
    # Take first action (should be MANA_CHARGE)
    action = actions[0]
    print(f"\nExecuting action: type={action.type}, src_iid={action.source_instance_id}, card_id={action.card_id}")
    
    # Get hand size before
    hand_size_before = len(state.get_zone(0, dm_ai_module.Zone.HAND))
    mana_size_before = len(state.get_zone(0, dm_ai_module.Zone.MANA))
    print(f"Before: hand={hand_size_before}, mana={mana_size_before}")
    
    # Execute action via GameInstance.resolve_action()
    try:
        gi.resolve_action(action)
        print("resolve_action() completed successfully")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Get hand/mana size after
    hand_size_after = len(state.get_zone(0, dm_ai_module.Zone.HAND))
    mana_size_after = len(state.get_zone(0, dm_ai_module.Zone.MANA))
    print(f"After: hand={hand_size_after}, mana={mana_size_after}")
    
    if hand_size_after < hand_size_before and mana_size_after > mana_size_before:
        print("\n✓ SUCCESS: Card moved from hand to mana!")
    else:
        print("\n✗ FAILURE: Card did not move")
        
    # Check phase progression
    print(f"Phase after action: {state.current_phase}")
