import pytest
from dm_ai_module import GameInstance, CommandSystem, CardStub

def test_select_target_and_destroy_logic():
    # 1. Initialize Game
    game = GameInstance()
    state = game.state
    state.setup_test_duel() # Ensure clean state

    # 2. Setup Board
    # Player 0 (Self) has 2 creatures in Battle Zone
    p0 = state.players[0]
    c1 = CardStub(1001, 1) # ID 1001, Instance 1
    c2 = CardStub(1002, 2) # ID 1002, Instance 2
    p0.battle_zone.append(c1)
    p0.battle_zone.append(c2)

    # 3. Create SELECT_TARGET Action
    # Select 1 creature from Self (Player 0) and store in "selected_var"
    select_cmd = {
        "type": "SELECT_TARGET",
        "target_group": "PLAYER_SELF",
        "amount": 1,
        "output_value_key": "selected_var"
    }

    # Execute Selection
    # Note: CommandSystem.execute_command(state, cmd, source_id, player_id)
    CommandSystem.execute_command(state, select_cmd, source_id=0, player_id=0)

    # Verify Selection
    assert "selected_var" in state.execution_context.variables
    selected_ids = state.execution_context.variables["selected_var"]
    assert len(selected_ids) == 1
    assert selected_ids[0] == 1 # Should select first one (c1) by default shim logic

    # 4. Create DESTROY Action linked to "selected_var"
    destroy_cmd = {
        "type": "DESTROY",
        "input_value_key": "selected_var"
    }

    # Execute Destruction
    CommandSystem.execute_command(state, destroy_cmd, source_id=0, player_id=0)

    # 5. Verify Result
    # c1 should be in graveyard, c2 should be in battle zone
    assert len(p0.battle_zone) == 1
    assert p0.battle_zone[0].instance_id == 2
    assert len(p0.graveyard) == 1
    assert p0.graveyard[0].instance_id == 1

if __name__ == "__main__":
    try:
        test_select_target_and_destroy_logic()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
