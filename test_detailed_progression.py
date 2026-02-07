"""Detailed test to track phase progression."""
import sys
sys.path.insert(0, '.')
from dm_toolkit.gui.game_session import GameSession

step_count = 0

def log(msg):
    global step_count
    print(f"[{step_count}] {msg}")

# Create session
session = GameSession(
    callback_log=log,
    callback_update_ui=lambda: None,
    callback_action_executed=None
)

session.player_modes = {0: 'AI', 1: 'AI'}

print("=== Initializing game ===")
session.initialize_game(seed=42)
print(f"Initial: turn={session.gs.turn_number}, phase={session.gs.current_phase}\n")

# Run 10 steps with detailed logging
for i in range(10):
    step_count = i
    
    # Get state before
    turn_before = session.gs.turn_number
    phase_before = session.gs.current_phase
    from dm_toolkit.engine.compat import EngineCompat
    player_before = EngineCompat.get_active_player_id(session.gs)
    
    print(f"\n=== Step {i+1} ===")
    print(f"Before: turn={turn_before}, phase={phase_before}, player={player_before}")
    
    # Execute step
    session.step_game()
    
    # Get state after
    turn_after = session.gs.turn_number
    phase_after = session.gs.current_phase
    player_after = EngineCompat.get_active_player_id(session.gs)
    
    print(f"After:  turn={turn_after}, phase={phase_after}, player={player_after}")
    
    # Detect changes
    if turn_after != turn_before:
        print(f"  → TURN CHANGED: {turn_before} → {turn_after}")
    if phase_after != phase_before:
        print(f"  → PHASE CHANGED: {phase_before} → {phase_after}")
    if player_after != player_before:
        print(f"  → PLAYER CHANGED: {player_before} → {player_after}")
    
    # Check if stuck
    if phase_after == phase_before and turn_after == turn_before and player_after == player_before:
        print(f"  WARNING: No progression!")

print("\n=== Test complete ===")
