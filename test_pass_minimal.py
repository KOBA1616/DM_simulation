#!/usr/bin/env python
"""Test PASS generation - minimal output version"""
import sys
sys.path.insert(0, '.')

try:
    import dm_ai_module as module
    from dm_toolkit.gui.game_session import GameSession
    
    session = GameSession(callback_log=lambda m: None, callback_update_ui=lambda: None, callback_action_executed=None)
    session.player_modes = {0: 'AI', 1: 'AI'}
    session.initialize_game(seed=42)
    
    gs = session.gs
    game_inst = session.game_instance
    card_db = session.native_card_db
    
    found = False
    for i in range(30):
        phase = int(gs.current_phase)
        if phase == 3:  # MAIN
            actions = module.IntentGenerator.generate_legal_actions(gs, card_db)
            action_types = [int(a.type) for a in actions]
            has_pass = 0 in action_types
            print(f"{i}: Phase 3 - {len(actions)} actions - has_PASS={has_pass} - types={action_types[:3]}")
            found = True
            break
        
        game_inst.step()
    
    if not found:
        print("Did not reach MAIN phase")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
