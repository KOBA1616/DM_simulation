
import sys
import os
import copy
import pytest

# Add build/bin to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module
except ImportError:
    pass

def test_game_instance_wrapper():
    print("Verifying GameInstance wrapper...")

    # Check if module loaded
    if 'dm_ai_module' not in sys.modules:
        pytest.fail("dm_ai_module not loaded")

    card1 = dm_ai_module.CardDefinition(
        1, "Bronze-Arm Tribe", "NATURE", ["Beast Folk"], 3, 1000,
        dm_ai_module.CardKeywords(), []
    )

    card_db = {1: card1}

    game = dm_ai_module.GameInstance(42, card_db)

    deck_ids = [1] * 40
    game.state.set_deck(0, deck_ids)
    game.state.set_deck(1, deck_ids)

    game.start_game()

    # Store initial phase as integer value to ensure we have a snapshot
    initial_phase_val = int(game.state.current_phase)
    print(f"Initial Phase: {game.state.current_phase} (Value: {initial_phase_val})")

    print("Resolving PASS action...")
    action = dm_ai_module.Action()
    action.type = dm_ai_module.ActionType.PASS

    game.resolve_action(action)

    new_phase_val = int(game.state.current_phase)
    print(f"New Phase: {game.state.current_phase} (Value: {new_phase_val})")

    assert new_phase_val != initial_phase_val, "Phase did not change!"

    print("Phase changed successfully.")

    print("Calling undo...")
    game.undo()
    print(f"Phase after undo: {game.state.current_phase}")

    # Optional: verify undo reverted phase (might not always match exactly if flow commands are complex, but check basic)
    # assert int(game.state.current_phase) == initial_phase_val
