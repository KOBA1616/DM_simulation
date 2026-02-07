#!/usr/bin/env python
"""Simple test to verify attack actions are generated when possible"""
import sys
sys.path.insert(0, '.')

from dm_toolkit.gui.game_session import GameSession

session = GameSession(
    callback_log=lambda msg: print(f"[LOG] {msg}"),
    callback_update_ui=lambda: None,
    callback_action_executed=None
)

session.player_modes = {0: 'AI', 1: 'AI'}
session.initialize_game(seed=42)

gs = session.gs
game_instance = session.game_instance

print("=== Starting game ===")
print(f"Turn: {gs.turn_number}, Phase: {gs.current_phase}")
print(f"P0: Hand={len(gs.players[0].hand)} MZ={len(gs.players[0].mana_zone)} BZ={len(gs.players[0].battle_zone)}")
print(f"P1: Hand={len(gs.players[1].hand)} MZ={len(gs.players[1].mana_zone)} BZ={len(gs.players[1].battle_zone)}")

# Fast forward until we reach ATTACK phase with a creature
max_steps = 100
for step_num in range(max_steps):
    if gs.game_over:
        print("Game over")
        break
    
    current_phase = int(gs.current_phase)
    active_pid = gs.active_player_id
    
    # Print state when phase changes or a creature appears
    p0_bz = len(gs.players[0].battle_zone)
    p1_bz = len(gs.players[1].battle_zone)
    
    print(f"\nStep {step_num}: Turn {gs.turn_number}, Phase {current_phase}, Active P{active_pid}")
    print(f"  P0: BZ={p0_bz}, P1: BZ={p1_bz}")
    
    # Check if we have an ATTACK phase with creatures
    if current_phase == 4 and active_pid == 0:  # ATTACK phase, Player 0 active
        if len(gs.players[0].battle_zone) > 0:
            print(f"  >>> IDEAL CONDITION: ATTACK phase with {len(gs.players[0].battle_zone)} creature(s) in active player BZ!")
            # Generate actions to see if attacks are possible
            from dm_toolkit.commands import generate_legal_commands
            cmds = generate_legal_commands(gs, session.card_db)
            print(f"  >>> Generated {len(cmds)} actions:")
            for i, cmd in enumerate(cmds[:5]):
                print(f"       Action {i}: {cmd}")
            if len(cmds) > 5:
                print(f"       ... and {len(cmds)-5} more")
            
            # Count attack actions
            attack_count = sum(1 for cmd in cmds if cmd.get('intent') in ['ATTACK_PLAYER', 'ATTACK_CREATURE'])
            print(f"  >>> Attack actions count: {attack_count}")
            
            if attack_count == 0 and len(cmds) > 0:
                print("  !!! BUG: ATTACK phase with creatures but NO attack actions generated")
                break
    
    try:
        success = game_instance.step()
        if not success:
            print(f"  step() returned false")
    except Exception as e:
        print(f"  Error: {e}")
        break

print(f"\nFinal: Turn {gs.turn_number}, Phase {gs.current_phase}")
print("Done")
