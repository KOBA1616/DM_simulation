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
from dm_toolkit import commands_v2 as commands
actions_mana = commands.generate_legal_commands(gs, cdb, strict=False)
print(f"Actions: {len(actions_mana)}")
for i, a in enumerate(actions_mana[:10]):
    print(f"  {i}: {a.type}")

# If any actions available, try executing the first using command-first API
if actions_mana:
    try:
        dm_ai_module.CommandSystem.execute_command(gs, actions_mana[0], player_id=gs.active_player_id, ctx=cdb)
    except Exception:
        # Best-effort fallback to dict-style command
        try:
            cmd = actions_mana[0] if isinstance(actions_mana[0], dict) else {'type': getattr(actions_mana[0], 'type', None), 'source_instance_id': getattr(actions_mana[0], 'source_instance_id', None), 'card_id': getattr(actions_mana[0], 'card_id', None)}
            dm_ai_module.CommandSystem.execute_command(gs, cmd, player_id=gs.active_player_id, ctx=cdb)
        except Exception:
            pass
    print(f"\nExecuted first action, mana zone now: {len(gs.players[0].mana_zone)} cards")

    # Regenerate actions
    actions_after = commands.generate_legal_commands(gs, cdb, strict=False)
    print(f"Actions after execution: {len(actions_after) if actions_after is not None else 0}")
    for i, a in enumerate((actions_after or [])[:10]):
        print(f"  {i}: {getattr(a, 'type', None)}")

# Now try MAIN phase
print(f"\n=== Forcing MAIN Phase ===")
gs.current_phase = dm_ai_module.Phase.MAIN
actions_main = commands.generate_legal_commands(gs, cdb, strict=False)
print(f"MAIN phase actions: {len(actions_main) if actions_main is not None else 0}")
for i, a in enumerate((actions_main or [])[:10]):
    print(f"  {i}: {getattr(a, 'type', None)}")
