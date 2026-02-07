#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from dm_toolkit.gui.game_session import GameSession

session = GameSession(
    callback_log=lambda m: None, 
    callback_update_ui=lambda: None, 
    callback_action_executed=None
)
session.player_modes = {0: 'AI', 1: 'AI'}
session.initialize_game(seed=42)

phase_names = {0: "START", 1: "DRAW", 2: "MANA", 3: "MAIN", 4: "ATTACK", 5: "BLOCK", 6: "END"}

print("Searching for ATTACK phase...\n")

for step in range(50):
    gs = session.gs
    phase = int(gs.current_phase)
    phase_name = phase_names.get(phase, f"UNKNOWN({phase})")
    
    p0_bz = len(gs.players[0].battle_zone)
    p1_bz = len(gs.players[1].battle_zone)
    
    if phase == 4:  # ATTACK phase
        print(f"Step {step}: ATTACK PHASE REACHED!")
        print(f"  Turn: {gs.turn_number}")
        print(f"  Active player: {gs.active_player_id}")
        print(f"  P0 battle_zone: {p0_bz}")
        if p0_bz > 0:
            for i, c in enumerate(gs.players[0].battle_zone):
                print(f"    P0[{i}]: id={c.card_id} tapped={c.is_tapped} sick={c.summoning_sickness} turn_played={c.turn_played}")
        print(f"  P1 battle_zone: {p1_bz}")
        if p1_bz > 0:
            for i, c in enumerate(gs.players[1].battle_zone):
                print(f"    P1[{i}]: id={c.card_id} tapped={c.is_tapped} sick={c.summoning_sickness} turn_played={c.turn_played}")
        print()
    
    try:
        session.step_game()
    except:
        break
        
    if gs.game_over:
        break
