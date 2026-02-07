#!/usr/bin/env python
"""
Debug: Check battle zone contents at each phase
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

print("=== Battle Zone Debug ===\n")

# Step through game
for step in range(15):
    gs = session.gs
    phase = int(gs.current_phase)
    p0_bz = len(gs.players[0].battle_zone)
    p1_bz = len(gs.players[1].battle_zone)
    
    phase_names = {
        0: "START", 1: "DRAW", 2: "MANA", 3: "MAIN", 
        4: "ATTACK", 5: "BLOCK", 6: "END"
    }
    phase_name = phase_names.get(phase, f"UNKNOWN({phase})")
    
    print(f"Step {step}: Phase={phase_name} Turn={gs.turn_number}")
    print(f"  P0 battle_zone: {p0_bz} creatures")
    print(f"  P1 battle_zone: {p1_bz} creatures")
    
    # If we have creatures, show details
    if p0_bz > 0:
        for i, c in enumerate(gs.players[0].battle_zone):
            print(f"    P0[{i}]: card_id={c.card_id} tapped={c.is_tapped} sick={c.summoning_sickness}")
    
    if p1_bz > 0:
        for i, c in enumerate(gs.players[1].battle_zone):
            print(f"    P1[{i}]: card_id={c.card_id} tapped={c.is_tapped} sick={c.summoning_sickness}")
    
    print()
    
    # Step game
    try:
        session.step_game()
    except Exception as e:
        print(f"Error at step {step}: {e}")
        break
    
    if gs.game_over:
        print(f"Game over at step {step}")
        break
