"""Simple test to debug GUI game session."""
import sys
sys.path.insert(0, '.')
from dm_toolkit.gui.game_session import GameSession

# Minimal logger
def log(msg):
    print(f"[LOG] {msg}")

# Create session
session = GameSession(
    callback_log=log,
    callback_update_ui=lambda: None,
    callback_action_executed=None
)

# Set player modes
session.player_modes = {0: 'AI', 1: 'AI'}

# Initialize
print("=== Initializing game ===")
session.initialize_game(seed=42)

print(f"\n=== Initial state: phase={session.gs.current_phase}, turn={session.gs.turn_number} ===\n")

# Run 3 steps
for i in range(3):
    print(f"\n--- Step {i+1} ---")
    session.step_game()
    print(f"After step: phase={session.gs.current_phase}, turn={session.gs.turn_number}, is_processing={session.is_processing}")
    
print("\n=== Test complete ===")
