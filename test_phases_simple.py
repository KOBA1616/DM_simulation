import dm_ai_module

# Load cards
cdb = dm_ai_module.JsonLoader.load_cards('data/cards.json')

# Setup game
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()
gs.set_deck(0, [1]*40)
gs.set_deck(1, [1]*40)

# Start game
dm_ai_module.PhaseManager.start_game(gs, cdb)
dm_ai_module.PhaseManager.fast_forward(gs, cdb)

print(f"Phase: {gs.current_phase}, Turn: {gs.turn_number}")
print(f"P0 hand: {len(gs.players[0].hand)}")
print(f"P0 mana zone: {len(gs.players[0].mana_zone)}")

# Generate actions in current phase (MANA)
print(f"\n=== Current phase ({gs.current_phase}) actions ===")
actions = dm_ai_module.IntentGenerator.generate_legal_actions(gs, cdb)
print(f"Total actions: {len(actions)}")
for i, a in enumerate(actions[:10]):
    print(f"  {i}: {a.type}")

# If we have MANA_CHARGE, execute it
if len(actions) > 0:
    first_action = actions[0]
    print(f"\n=== Executing first action: {first_action.type} ===")
    result = dm_ai_module.CommandExecutor.execute_action(gs, first_action, cdb)
    print(f"Result: {result}")
    print(f"P0 mana zone after: {len(gs.players[0].mana_zone)}")
    
    # Check actions again
    print(f"\n=== Actions after execution ===")
    actions2 = dm_ai_module.IntentGenerator.generate_legal_actions(gs, cdb)
    print(f"Total actions: {len(actions2)}")
    for i, a in enumerate(actions2[:10]):
        print(f"  {i}: {a.type}")
    
    # Execute PASS if available
    if len(actions2) > 0 and actions2[0].type == dm_ai_module.PlayerIntent.PASS:
        print(f"\n=== Executing PASS to exit MANA phase ===")
        dm_ai_module.CommandExecutor.execute_action(gs, actions2[0], cdb)
        print(f"New phase: {gs.current_phase}")
        
        # Check MAIN phase actions
        if gs.current_phase == dm_ai_module.Phase.MAIN:
            print(f"\n=== MAIN phase actions ===")
            actions3 = dm_ai_module.IntentGenerator.generate_legal_actions(gs, cdb)
            print(f"Total actions: {len(actions3)}")
            for i, a in enumerate(actions3[:10]):
                print(f"  {i}: {a.type}")
