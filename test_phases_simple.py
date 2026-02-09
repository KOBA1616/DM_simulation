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
print(f"\n=== Current phase ({gs.current_phase}) commands ===")
from dm_toolkit import commands_v2 as commands
import dm_ai_module as _dm

def _get_legal(gs, card_db):
    try:
        cmds = commands.generate_legal_commands(gs, card_db, strict=False) or []
    except Exception:
        cmds = []
    if not cmds:
        try:
            try:
                cmds = commands.generate_legal_commands(gs, card_db, strict=False) or []
            except Exception:
                try:
                    cmds = commands.generate_legal_commands(gs, card_db) or []
                except Exception:
                    cmds = []
        except Exception:
            cmds = []
    return cmds

actions = _get_legal(gs, cdb)
print(f"Total commands: {len(actions) if actions is not None else 0}")
for i, a in enumerate((actions or [])[:10]):
    print(f"  {i}: {getattr(a, 'type', None)}")

# If we have MANA_CHARGE, execute it
    if len(actions) > 0:
    first_action = actions[0]
    print(f"\n=== Executing first action: {getattr(first_action, 'type', None)} ===")
    try:
        dm_ai_module.CommandSystem.execute_command(gs, first_action, player_id=gs.active_player_id, ctx=cdb)
    except Exception:
        try:
            dm_ai_module.CommandSystem.execute_command(gs, dict(type=getattr(first_action, 'type', None), source_instance_id=getattr(first_action, 'source_instance_id', None), card_id=getattr(first_action, 'card_id', None)), player_id=gs.active_player_id, ctx=cdb)
        except Exception:
            pass
    print(f"P0 mana zone after: {len(gs.players[0].mana_zone)}")
    
    # Check actions again
    print(f"\n=== Actions after execution ===")
    actions2 = commands.generate_legal_commands(gs, cdb, strict=False)
    print(f"Total actions: {len(actions2)}")
    for i, a in enumerate(actions2[:10]):
        print(f"  {i}: {a.type}")
    
    # Execute PASS if available
    if len(actions2) > 0 and getattr(actions2[0], 'type', None) == getattr(dm_ai_module, 'PlayerIntent', None).PASS if hasattr(dm_ai_module, 'PlayerIntent') else None:
        print(f"\n=== Executing PASS to exit MANA phase ===")
        try:
            dm_ai_module.CommandSystem.execute_command(gs, actions2[0], player_id=gs.active_player_id, ctx=cdb)
        except Exception:
            try:
                dm_ai_module.CommandSystem.execute_command(gs, dict(type=getattr(actions2[0], 'type', None)), player_id=gs.active_player_id, ctx=cdb)
            except Exception:
                pass
        print(f"New phase: {gs.current_phase}")
        
        # Check MAIN phase actions
        if gs.current_phase == dm_ai_module.Phase.MAIN:
            print(f"\n=== MAIN phase actions ===")
            actions3 = commands.generate_legal_commands(gs, cdb, strict=False)
            print(f"Total actions: {len(actions3) if actions3 is not None else 0}")
            for i, a in enumerate((actions3 or [])[:10]):
                print(f"  {i}: {getattr(a, 'type', None)}")
