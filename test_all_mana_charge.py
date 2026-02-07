"""Test executing ALL MANA_CHARGE actions to reach MAIN_PHASE."""
import sys
sys.path.insert(0, '.')
import dm_ai_module

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
dm_ai_module.PhaseManager.fast_forward(gs, card_db)
gs = game_instance.state

print(f"After start_game: phase={gs.current_phase}, turn={gs.turn_number}")

# Execute ALL MANA_CHARGE actions
mana_count = 0
max_iterations = 20  # Safety limit
for iteration in range(max_iterations):
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
    
    if not actions:
        print(f"No actions available. Fast-forwarding...")
        dm_ai_module.PhaseManager.fast_forward(gs, card_db)
        gs = game_instance.state
        continue
    
    # Find MANA_CHARGE action
    mana_actions = [a for a in actions if int(a.type) == 1]  # ActionType.MANA_CHARGE
    
    if not mana_actions:
        print(f"No MANA_CHARGE actions. Current phase: {gs.current_phase}")
        break
    
    # Execute first MANA_CHARGE
    action = mana_actions[0]
    print(f"[{iteration}] Executing MANA_CHARGE card_id={action.card_id}, phase={gs.current_phase}")
    
    game_instance.resolve_action(action)
    gs = game_instance.state
    mana_count += 1
    
    # Fast-forward to next action point
    dm_ai_module.PhaseManager.fast_forward(gs, card_db)
    gs = game_instance.state
    
    if gs.current_phase != dm_ai_module.Phase.MANA:
        print(f"Phase changed to: {gs.current_phase}")
        break

print(f"\nFinal state:")
print(f"  Phase: {gs.current_phase}")
print(f"  Turn: {gs.turn_number}")
print(f"  Current player: {gs.current_player}")
print(f"  Mana used: {mana_count}")

# Check MAIN_PHASE actions
if gs.current_phase == dm_ai_module.Phase.MAIN:
    print(f"\n[SUCCESS] Reached MAIN_PHASE!")
    main_actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
    print(f"  MAIN_PHASE actions: {len(main_actions)}")
    
    play_actions = [a for a in main_actions if int(a.type) == 2]  # ActionType.PLAY
    print(f"  PLAY actions: {len(play_actions)}")
    
    if play_actions:
        print(f"  Example PLAY action: card_id={play_actions[0].card_id}")
else:
    print(f"\n[FAILED] Still in {gs.current_phase}, not MAIN_PHASE")
