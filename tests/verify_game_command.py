import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'bin'))
import dm_ai_module

def test_command_execution():
    print("Initializing GameState...")
    state = dm_ai_module.GameState(1000)

    # Setup players
    # add_test_card_to_battle(player, card_id, inst_id, tapped, sick)
    state.add_test_card_to_battle(0, 100, 10, False, True)

    # Check initial state
    card = state.get_card_instance(10)
    assert card is not None, "Card should exist"
    assert card.is_tapped == False, "Card should be untapped"

    print("Executing MutateCommand (TAP)...")
    # MutateCommand(id, type, val, str)
    cmd = dm_ai_module.MutateCommand(10, dm_ai_module.MutationType.TAP, 0, "")
    state.execute_command(cmd)

    assert card.is_tapped == True, "Card should be tapped after command"

    print("Executing Undo...")
    state.undo_last_command()
    assert card.is_tapped == False, "Card should be untapped after undo"

    print("Command Execution and Undo Verified!")

    # Test Transition
    print("Testing TransitionCommand...")
    # TransitionCommand(instance_id, from, to, owner, dest_idx)
    # Move from Battle to Hand
    cmd_move = dm_ai_module.TransitionCommand(10, dm_ai_module.Zone.BATTLE, dm_ai_module.Zone.HAND, 0, -1)
    state.execute_command(cmd_move)

    # Verify move
    card_in_battle = False
    for c in state.players[0].battle_zone:
        if c.instance_id == 10: card_in_battle = True
    assert not card_in_battle, "Card should not be in battle zone"

    card_in_hand = False
    for c in state.players[0].hand:
        if c.instance_id == 10: card_in_hand = True
    assert card_in_hand, "Card should be in hand"

    # Undo move
    state.undo_last_command()

    card_in_battle = False
    for c in state.players[0].battle_zone:
        if c.instance_id == 10: card_in_battle = True
    assert card_in_battle, "Card should be back in battle zone"

    print("Transition Execution and Undo Verified!")

if __name__ == "__main__":
    test_command_execution()
