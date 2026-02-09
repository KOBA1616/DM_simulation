"""Test if game_instance.state reflects C++ internal state after resolve_action."""
import sys
sys.path.insert(0, '.')
import dm_ai_module

seed = 42
card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
game_instance = dm_ai_module.GameInstance(seed, card_db)
gs = game_instance.state

gs.setup_test_duel()
deck = [1,2,3,4,5,6,7,8,9,10]*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)

dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)

print(f"Initial state:")
print(f"  gs.turn_number: {gs.turn_number}")
print(f"  gs.current_phase: {gs.current_phase}")

# Get actions and execute ONE MANA_CHARGE with PASS
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

actions = _get_legal(gs, card_db)
mana_actions = [a for a in actions if int(a.type) == 1]  # MANA_CHARGE
pass_actions = [a for a in actions if int(a.type) == 0]  # PASS

if mana_actions:
    print(f"\nExecuting 1 MANA_CHARGE...")
    game_instance.resolve_action(mana_actions[0])
    
    # Check gs directly (without re-fetching)
    print(f"After resolve_action (without re-fetch):")
    print(f"  gs.turn_number: {gs.turn_number}")
    print(f"  gs.current_phase: {gs.current_phase}")
    print(f"  gs.players[0].mana_zone length: {len(gs.players[0].mana_zone)}")
    
    # Re-fetch from game_instance
    gs_new = game_instance.state
    print(f"\nAfter re-fetch (gs_new = game_instance.state):")
    print(f"  gs_new.turn_number: {gs_new.turn_number}")
    print(f"  gs_new.current_phase: {gs_new.current_phase}")
    print(f"  gs_new.players[0].mana_zone length: {len(gs_new.players[0].mana_zone)}")
    
    print(f"\nAre they the same object? {gs is gs_new}")
    
    # Now execute PASS to advance phase
    actions2 = _get_legal(gs, card_db)
    pass_actions2 = [a for a in (actions2 or []) if str(getattr(a, 'type', '')).isdigit() and int(a.type) == 0]
    
    if pass_actions2:
        print(f"\n=== Executing PASS to advance phase ===")
        game_instance.resolve_action(pass_actions2[0])
        dm_ai_module.PhaseManager.fast_forward(gs, card_db)
        
        print(f"After PASS + fast_forward:")
        print(f"  gs.turn_number: {gs.turn_number}")
        print(f"  gs.current_phase: {gs.current_phase}")
        
        gs_after = game_instance.state
        print(f"\nAfter re-fetch:")
        print(f"  gs_after.turn_number: {gs_after.turn_number}")
        print(f"  gs_after.current_phase: {gs_after.current_phase}")
