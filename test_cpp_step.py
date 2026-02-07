"""Test C++ step() method for automated game progression."""
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
print(f"Turn: {gs.turn_number}, Phase: {gs.current_phase}")
print(f"\n=== Running 10 C++ steps ===\n")

for i in range(10):
    turn_before = gs.turn_number
    phase_before = gs.current_phase
    
    # Call C++ step() - it handles everything
    success = game_instance.step()
    
    turn_after = gs.turn_number
    phase_after = gs.current_phase
    
    print(f"Step {i+1}: success={success}, Turn {turn_before}→{turn_after}, Phase {phase_before}→{phase_after}")
    
    if not success:
        print(f"  => step() returned False, stopping")
        break
    
    if gs.game_over:
        print(f"  => Game Over!")
        break

print(f"\n=== Final State ===")
print(f"Turn: {gs.turn_number}, Phase: {gs.current_phase}")
print(f"Game Over: {gs.game_over}")

# Check current state
p0 = gs.players[0]
p1 = gs.players[1]
print(f"\nP0: Hand={len(p0.hand)}, Mana={len(p0.mana_zone)}, Battle={len(p0.battle_zone)}")
print(f"P1: Hand={len(p1.hand)}, Mana={len(p1.mana_zone)}, Battle={len(p1.battle_zone)}")
