
import pytest
import dm_ai_module as dm
from dm_ai_module import GameState, CommandType, TransitionCommand, MutateCommand, FlowCommand, QueryCommand, DecideCommand, Zone, Phase, MutationType, FlowType

def test_transition_command():
    state = GameState(100)
    # Setup: Add a card to HAND
    player_id = 0
    card_id = 1
    instance_id = 100
    state.add_card_to_hand(player_id, card_id, instance_id)

    assert len(state.players[player_id].hand) == 1
    assert len(state.players[player_id].mana_zone) == 0

    # Execute TransitionCommand: HAND -> MANA
    cmd = TransitionCommand(instance_id, Zone.HAND, Zone.MANA, player_id)
    cmd.execute(state)

    assert len(state.players[player_id].hand) == 0
    assert len(state.players[player_id].mana_zone) == 1
    assert state.players[player_id].mana_zone[0].instance_id == instance_id

    # Execute Invert (Undo)
    cmd.invert(state)

    assert len(state.players[player_id].hand) == 1
    assert len(state.players[player_id].mana_zone) == 0
    assert state.players[player_id].hand[0].instance_id == instance_id

def test_mutate_command_tap():
    state = GameState(100)
    # Setup: Add a card to BATTLE
    player_id = 0
    card_id = 1
    instance_id = 200
    state.add_test_card_to_battle(player_id, card_id, instance_id, False, False)

    card = state.get_card_instance(instance_id)
    assert not card.is_tapped

    # Execute MutateCommand: TAP
    cmd = MutateCommand(instance_id, MutationType.TAP)
    cmd.execute(state)

    card = state.get_card_instance(instance_id)
    assert card.is_tapped

    # Execute Invert (Undo)
    cmd.invert(state)

    card = state.get_card_instance(instance_id)
    assert not card.is_tapped

def test_mutate_command_power():
    state = GameState(100)
    player_id = 0
    card_id = 1
    instance_id = 300
    state.add_test_card_to_battle(player_id, card_id, instance_id, False, False)

    card = state.get_card_instance(instance_id)
    assert card.power_mod == 0

    # Execute MutateCommand: POWER_MOD +1000
    cmd = MutateCommand(instance_id, MutationType.POWER_MOD, 1000)
    cmd.execute(state)

    card = state.get_card_instance(instance_id)
    assert card.power_mod == 1000

    # Execute Invert
    cmd.invert(state)

    card = state.get_card_instance(instance_id)
    assert card.power_mod == 0

def test_flow_command():
    state = GameState(100)
    state.current_phase = Phase.MAIN

    # Execute FlowCommand: Change to ATTACK
    # Note: Enum values are integers in pybind, usually. But let's pass the enum.
    # The constructor takes (FlowType, int).
    cmd = FlowCommand(FlowType.PHASE_CHANGE, int(Phase.ATTACK))
    cmd.execute(state)

    assert state.current_phase == Phase.ATTACK

    # Invert
    cmd.invert(state)

    assert state.current_phase == Phase.MAIN

def test_query_command():
    state = GameState(100)
    assert not state.waiting_for_user_input

    cmd = QueryCommand("SELECT_TARGET", [1, 2, 3], {"min": 1, "max": 1})
    cmd.execute(state)

    assert state.waiting_for_user_input
    assert state.pending_query.query_type == "SELECT_TARGET"
    assert state.pending_query.valid_target_ids == [1, 2, 3]

    cmd.invert(state)

    assert not state.waiting_for_user_input
