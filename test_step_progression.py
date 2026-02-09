import dm_ai_module
from dm_toolkit import commands_v2 as commands

# Load cards
cdb = dm_ai_module.JsonLoader.load_cards('data/cards.json')

# Create GameInstance
gi = dm_ai_module.GameInstance(42, cdb)
gs = gi.state

# Setup game
gs.setup_test_duel()
gs.set_deck(0, [1]*40)
gs.set_deck(1, [1]*40)

# Start game
dm_ai_module.PhaseManager.start_game(gs, cdb)
dm_ai_module.PhaseManager.fast_forward(gs, cdb)

print("=== Initial State ===")
print(f"Phase: {gs.current_phase}, Turn: {gs.turn_number}")
print(f"P0 hand: {len(gs.players[0].hand)}")
print(f"P0 mana zone: {len(gs.players[0].mana_zone)}")

# Step 1: MANA_CHARGE
print("\n=== Step 1: Execute MANA_CHARGE ===")
success1 = gi.step()
print(f"Success: {success1}")
print(f"Phase: {gs.current_phase}, P0 mana: {len(gs.players[0].mana_zone)}")

# Step 2: PASS (exit MANA)
print("\n=== Step 2: Execute PASS to exit MANA ===")
success2 = gi.step()
print(f"Success: {success2}")
print(f"Phase: {gs.current_phase}")

# Step 3: Check MAIN phase
if gs.current_phase == dm_ai_module.Phase.MAIN:
    print("\n=== Now in MAIN Phase ===")
    actions = commands.generate_legal_commands(gs, cdb, strict=False)
    print(f"Available actions: {len(actions)}")
    for i, a in enumerate(actions[:10]):
        print(f"  {i}: {a.type}")
    
    # Try one step in MAIN
    if len(actions) > 0:
        print("\n=== Step in MAIN phase ===")
        success3 = gi.step()
        print(f"Success: {success3}")
        print(f"Phase after: {gs.current_phase}")
else:
    print(f"\nERROR: Not in MAIN phase, still in {gs.current_phase}")
