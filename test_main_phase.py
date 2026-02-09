import dm_ai_module

# Load cards
cdb = dm_ai_module.JsonLoader.load_cards('data/cards.json')

# Setup game
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()
gs.set_deck(0, [1]*40)
gs.set_deck(1, [1]*40)

# Start game and advance to MAIN phase
dm_ai_module.PhaseManager.start_game(gs, cdb)
dm_ai_module.PhaseManager.fast_forward(gs, cdb)

# Force to MAIN phase
gs.current_phase = dm_ai_module.Phase.MAIN

print(f"Phase: {gs.current_phase}, Turn: {gs.turn_number}")
print(f"P0 hand size: {len(gs.players[0].hand)}")

# Generate actions
from dm_toolkit import commands_v2 as commands
actions = commands.generate_legal_commands(gs, cdb, strict=False)
print(f"\nMAIN phase actions: {len(actions)}")

# Show first 20 actions
for i, a in enumerate(actions[:20]):
    print(f"  {i}: {a.type}")
