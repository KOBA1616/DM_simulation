#!/usr/bin/env python
"""
Test mana charge and zone transitions
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
print(f"Initial state:")
print(f"  P0: Hand={len(gs.players[0].hand)} MZ={len(gs.players[0].mana_zone)} BZ={len(gs.players[0].battle_zone)}")
print(f"  P1: Hand={len(gs.players[1].hand)} MZ={len(gs.players[1].mana_zone)} BZ={len(gs.players[1].battle_zone)}")
print()

# Step 3 times
for i in range(3):
    print(f"Step {i}: Phase={gs.current_phase}")
    phase_names = {0: "START", 1: "DRAW", 2: "MANA", 3: "MAIN", 4: "ATTACK"}
    phase_name = phase_names.get(int(gs.current_phase), "UNKNOWN")
    print(f"         {phase_name}")
    
    try:
        session.step_game()
    except Exception as e:
        print(f"Error: {e}")
        break
    
    print(f"  P0: Hand={len(gs.players[0].hand)} MZ={len(gs.players[0].mana_zone)} BZ={len(gs.players[0].battle_zone)}")
    print(f"  P1: Hand={len(gs.players[1].hand)} MZ={len(gs.players[1].mana_zone)} BZ={len(gs.players[1].battle_zone)}")
    print()

print("Done")
