
import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append('bin')

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure it is built.")
    sys.exit(1)

def test_transition_command():
    print("Testing TransitionCommand...")

    # Setup state
    state = dm_ai_module.GameState(100)

    # Add a card to HAND
    player_id = 0
    card_id = 1
    instance_id = 100

    state.add_card_to_hand(player_id, card_id, instance_id)

    # Verify initial state
    hand = state.players[player_id].hand
    assert len(hand) == 1
    assert hand[0].instance_id == instance_id

    # Create TransitionCommand: HAND -> MANA
    cmd = dm_ai_module.TransitionCommand(
        instance_id,
        dm_ai_module.Zone.HAND,
        dm_ai_module.Zone.MANA,
        player_id,
        -1
    )

    # Execute
    cmd.execute(state)

    # Verify execution
    hand = state.players[player_id].hand
    mana = state.players[player_id].mana_zone
    assert len(hand) == 0
    assert len(mana) == 1
    assert mana[0].instance_id == instance_id

    print("  Execution passed.")

    # Invert (Undo)
    cmd.invert(state)

    # Verify inversion
    hand = state.players[player_id].hand
    mana = state.players[player_id].mana_zone
    assert len(mana) == 0
    assert len(hand) == 1
    assert hand[0].instance_id == instance_id

    print("  Inversion passed.")

def test_transition_command_position():
    print("Testing TransitionCommand with position...")

    state = dm_ai_module.GameState(100)
    player_id = 0

    # Add 3 cards to HAND
    for i in range(3):
        state.add_card_to_hand(player_id, 1, 100 + i)

    # Move the middle card (101) to MANA
    cmd = dm_ai_module.TransitionCommand(
        101,
        dm_ai_module.Zone.HAND,
        dm_ai_module.Zone.MANA,
        player_id,
        -1
    )

    cmd.execute(state)

    hand = state.players[player_id].hand
    assert len(hand) == 2
    assert hand[0].instance_id == 100
    assert hand[1].instance_id == 102

    cmd.invert(state)

    hand = state.players[player_id].hand
    assert len(hand) == 3
    # Check if order is preserved (101 should be back at index 1)
    # The current implementation of invert uses original_index
    assert hand[0].instance_id == 100
    assert hand[1].instance_id == 101
    assert hand[2].instance_id == 102

    print("  Position preservation passed.")

if __name__ == "__main__":
    test_transition_command()
    test_transition_command_position()
    print("All tests passed.")
