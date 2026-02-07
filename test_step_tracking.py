"""Extremely detailed step tracking to find where state changes."""
import sys
sys.path.insert(0, '.')
from dm_toolkit.gui.game_session import GameSession
from dm_toolkit.engine.compat import EngineCompat

def get_state_snapshot(gs):
    """Get current state as dict."""
    return {
        'turn': gs.turn_number,
        'phase': str(gs.current_phase),
        'player': EngineCompat.get_active_player_id(gs)
    }

def log_state_change(label, before, after):
    """Log state changes."""
    changes = []
    for key in before:
        if before[key] != after[key]:
            changes.append(f"{key}: {before[key]} → {after[key]}")
    if changes:
        print(f"  [{label}] State changed: {', '.join(changes)}")
    else:
        print(f"  [{label}] No change")

# Create session
session = GameSession(
    callback_log=lambda msg: print(f"[LOG] {msg}"),
    callback_update_ui=lambda: None,
    callback_action_executed=None
)

session.player_modes = {0: 'AI', 1: 'AI'}

print("=== Initializing game ===")
s0 = get_state_snapshot(session.gs) if session.gs else None
session.initialize_game(seed=42)
s1 = get_state_snapshot(session.gs)
if s0:
    log_state_change("After initialize_game", s0, s1)
else:
    print(f"Initial state: {s1}")

print(f"\n=== Running 5 steps ===\n")

for i in range(5):
    print(f"\n--- Step {i+1} START ---")
    before_step = get_state_snapshot(session.gs)
    print(f"Before step_game(): {before_step}")
    
    # Manually call step_game() and track internal state changes
    session.step_game()
    
    after_step = get_state_snapshot(session.gs)
    print(f"After step_game(): {after_step}")
    log_state_change("step_game", before_step, after_step)
    print(f"--- Step {i+1} END ---")

print("\n=== Test complete ===")
