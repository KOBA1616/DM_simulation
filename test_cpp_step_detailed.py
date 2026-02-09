"""Test C++ step() method with detailed action logging."""
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

print(f"=== Initial State ===")
print(f"Turn: {gs.turn_number}, Phase: {gs.current_phase}, Active Player: {gs.active_player_id}\n")

for i in range(5):
    print(f"--- Step {i+1} ---")
    print(f"Before: Turn={gs.turn_number}, Phase={gs.current_phase}, Player={gs.active_player_id}")
    
    # Generate commands to see what's available
    from dm_toolkit import commands_v2 as commands
    actions = commands.generate_legal_commands(gs, card_db, strict=False)
    print(f"Commands available ({len(actions) if actions is not None else 0}):")
    for j, action in enumerate((actions or [])[:5]):  # Show first 5
        print(f"  {j+1}. {getattr(action, 'type', None)}")
    if len(actions) > 5:
        print(f"  ... and {len(actions)-5} more")
    
    # Call C++ step()
    success = game_instance.step()
    
    print(f"After:  Turn={gs.turn_number}, Phase={gs.current_phase}, Player={gs.active_player_id}, success={success}")
    print()
    
    if not success or gs.game_over:
        break

print(f"=== Final State ===")
print(f"Turn: {gs.turn_number}, Phase: {gs.current_phase}")
p0 = gs.players[0]
p1 = gs.players[1]
print(f"P0: Hand={len(p0.hand)}, Mana={len(p0.mana_zone)}, Battle={len(p0.battle_zone)}")
print(f"P1: Hand={len(p1.hand)}, Mana={len(p1.mana_zone)}, Battle={len(p1.battle_zone)}")
