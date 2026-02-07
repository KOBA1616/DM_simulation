import dm_ai_module

# Load cards
cdb = dm_ai_module.JsonLoader.load_cards('data/cards.json')

# Setup game
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()
gs.set_deck(0, [1]*40)
gs.set_deck(1, [1]*40)

# Start game and advance through MANA phase
dm_ai_module.PhaseManager.start_game(gs, cdb)
print(f"After start: Phase={gs.current_phase}, Turn={gs.turn_number}")

# Fast forward to after DRAW
dm_ai_module.PhaseManager.fast_forward(gs, cdb)
print(f"After fast_forward: Phase={gs.current_phase}, Turn={gs.turn_number}")

# Check mana stats  
print(f"\nTurn stats:")
print(f"  Mana charged by P0: {gs.turn_stats.mana_charged_by_player[0]}")
print(f"  Mana charged by P1: {gs.turn_stats.mana_charged_by_player[1]}")

# Check player states
print(f"\nPlayer 0:")
print(f"  Hand: {len(gs.players[0].hand)} cards")
print(f"  Mana zone: {len(gs.players[0].mana_zone)} cards")
print(f"  Battle zone: {len(gs.players[0].battle_zone)} creatures")

# Try to generate actions in MANA phase
print(f"\n=== MANA Phase Actions ===")
actions_mana = dm_ai_module.IntentGenerator.generate_legal_actions(gs, cdb)
print(f"Actions: {len(actions_mana)}")
for i, a in enumerate(actions_mana[:10]):
    print(f"  {i}: {a.type}")

# Execute MANA_CHARGE if available
if len(actions_mana) > 0 and actions_mana[0].type == dm_ai_module.PlayerIntent.MANA_CHARGE:
    result = dm_ai_module.CommandExecutor.execute_action(gs, actions_mana[0], cdb)
    print(f"\nExecuted MANA_CHARGE, result: {result}")
    print(f"Mana zone now: {len(gs.players[0].mana_zone)} cards")
    
    # Regenerate actions
    actions_after = dm_ai_module.IntentGenerator.generate_legal_actions(gs, cdb)
    print(f"Actions after MANA_CHARGE: {len(actions_after)}")
    for i, a in enumerate(actions_after[:10]):
        print(f"  {i}: {a.type}")

# Now try MAIN phase
print(f"\n=== Forcing MAIN Phase ===")
gs.current_phase = dm_ai_module.Phase.MAIN
actions_main = dm_ai_module.IntentGenerator.generate_legal_actions(gs, cdb)
print(f"MAIN phase actions: {len(actions_main)}")
for i, a in enumerate(actions_main[:10]):
    print(f"  {i}: {a.type}")
