"""Test if PhaseManager.fast_forward actually changes phase."""
import sys
sys.path.insert(0, '.')
import dm_ai_module
from dm_toolkit import commands_v2 as commands

seed = 42
card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
game_instance = dm_ai_module.GameInstance(seed, card_db)
gs = game_instance.state

gs.setup_test_duel()
deck = [1,2,3,4,5,6,7,8,9,10]*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)

dm_ai_module.PhaseManager.start_game(gs, card_db)
print(f"After start_game: phase={gs.current_phase}, turn={gs.turn_number}")

# Call fast_forward
print("\n=== Calling fast_forward ===")
dm_ai_module.PhaseManager.fast_forward(gs, card_db)
print(f"After fast_forward: phase={gs.current_phase}, turn={gs.turn_number}")

# Execute ONE MANA_CHARGE, then fast_forward
actions = commands.generate_legal_commands(gs, card_db, strict=False)
mana_actions = [a for a in actions if int(a.type) == 1]
if mana_actions:
    print(f"\n=== Executing MANA_CHARGE ===")
    action = mana_actions[0]
    print(f"Before: phase={gs.current_phase}, mana={len(gs.players[0].mana_zone)}")
    
    game_instance.resolve_action(action)
    print(f"After resolve_action: phase={gs.current_phase}, mana={len(gs.players[0].mana_zone)}")
    
    dm_ai_module.PhaseManager.fast_forward(gs, card_db)
    print(f"After fast_forward: phase={gs.current_phase}, mana={len(gs.players[0].mana_zone)}")
    
    # Check actions in current phase
    actions_after = commands.generate_legal_commands(gs, card_db, strict=False)
    print(f"\nActions available: {len(actions_after) if actions_after is not None else 0}")
    if actions_after:
        for i, a in enumerate((actions_after or [])[:5]):
            print(f"  [{i}] type={getattr(a, 'type', None)} card_id={getattr(a, 'card_id', None)}")
