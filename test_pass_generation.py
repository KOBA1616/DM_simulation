#!/usr/bin/env python
"""Minimal test to see if PASS is generated in MAIN phase"""
import sys
sys.path.insert(0, '.')

import dm_ai_module as module
from dm_toolkit.gui.game_session import GameSession

# Create session
session = GameSession(
    callback_log=lambda m: None,
    callback_update_ui=lambda: None,
    callback_action_executed=None
)
session.player_modes = {0: 'AI', 1: 'AI'}
session.initialize_game(seed=42)

gs = session.gs
game_inst = session.game_instance
card_db = session.native_card_db

# Fast-forward to MAIN phase where no cards can be played
for _ in range(20):
    phase = int(gs.current_phase)
    if phase == 3:  # MAIN phase (3)
        # Generate actions
        from dm_toolkit import commands_v2 as commands
        actions = commands.generate_legal_commands(gs, card_db, strict=False)
        print(f"Phase {phase} (MAIN): Generated {len(actions)} actions")
        for i, a in enumerate(actions[:5]):
            print(f"  Action {i}: type={int(a.type)}")
        
        # Check for PASS (type = 0)
        has_pass = any(int(a.type) == 0 for a in actions)
        print(f"  Has PASS: {has_pass}")
        
        if len(actions) > 0:
            break
    
    game_inst.step()

print("Done")
