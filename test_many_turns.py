#!/usr/bin/env python
"""
Test multiple turns to see mana accumulation
"""
import sys
sys.path.insert(0, '.')
from dm_toolkit.gui.game_session import GameSession

class Logger:
    def log_message(self, msg):
        pass

session = GameSession(
    callback_log=Logger().log_message, 
    callback_update_ui=lambda: None, 
    callback_action_executed=None
)
session.player_modes = {0: 'AI', 1: 'AI'}
session.initialize_game(seed=42)

gs = session.gs
p0 = gs.players[0]
p1 = gs.players[1]

print("Initial:")
print(f"  Turn: {gs.turn_number}, Phase: {gs.current_phase}")
print(f"  P0: Hand={len(p0.hand)} MZ={len(p0.mana_zone)} BZ={len(p0.battle_zone)}")
print(f"  P1: Hand={len(p1.hand)} MZ={len(p1.mana_zone)} BZ={len(p1.battle_zone)}")
print()

# Step many times
for step_num in range(15):
    try:
        session.step_game()
    except Exception as e:
        print(f"Error at step {step_num}: {e}")
        break
    
    print(f"Step {step_num+1}:")
    print(f"  Turn: {gs.turn_number}, Phase: {gs.current_phase}")
    print(f"  P0: Hand={len(p0.hand)} MZ={len(p0.mana_zone)} BZ={len(p0.battle_zone)}")
    print(f"  P1: Hand={len(p1.hand)} MZ={len(p1.mana_zone)} BZ={len(p1.battle_zone)}")
    
    if gs.game_over:
        print("Game over")
        break

print("Done")
