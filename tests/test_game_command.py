
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

def test_transition_stack():
    state = GameState(100)
    player_id = 0
    card_id = 1
    instance_id = 400
    state.add_card_to_hand(player_id, card_id, instance_id)

    # Move Hand -> Stack
    cmd = TransitionCommand(instance_id, Zone.HAND, Zone.STACK, player_id)
    cmd.execute(state)

    # Check stack zone (exposed to python?)
    # GameState exposes stack_zone as read-write vector of CardInstance if bound
    assert len(state.stack_zone) == 1
    assert state.stack_zone[0].instance_id == instance_id
    assert len(state.players[player_id].hand) == 0

    # Invert
    cmd.invert(state)
    assert len(state.stack_zone) == 0
    assert len(state.players[player_id].hand) == 1

def test_mutate_attack_state():
    state = GameState(100)

    # Verify defaults
    assert state.current_attack.source_instance_id == -1
    assert state.current_attack.target_instance_id == -1

    # Set Source
    cmd_src = MutateCommand(-1, MutationType.SET_ATTACK_SOURCE, 1001)
    cmd_src.execute(state)
    assert state.current_attack.source_instance_id == 1001

    # Set Target Creature
    cmd_tgt = MutateCommand(-1, MutationType.SET_ATTACK_TARGET, 2002)
    cmd_tgt.execute(state)
    assert state.current_attack.target_instance_id == 2002

    # Set Target Player
    cmd_ply = MutateCommand(-1, MutationType.SET_ATTACK_PLAYER, 1)
    cmd_ply.execute(state)
    assert state.current_attack.target_player == 1

    # Set Blocker
    cmd_blk = MutateCommand(-1, MutationType.SET_BLOCKER, 3003)
    cmd_blk.execute(state)
    assert state.current_attack.blocker_instance_id == 3003
    assert state.current_attack.is_blocked == True

    # Invert Blocker
    cmd_blk.invert(state)
    assert state.current_attack.blocker_instance_id == -1
    assert state.current_attack.is_blocked == False

    # Invert Player
    cmd_ply.invert(state)
    assert state.current_attack.target_player != 1 # Assuming default was 255 or 0, depending on constructor.
    # We should have checked default before.

    # Invert Target
    cmd_tgt.invert(state)
    assert state.current_attack.target_instance_id == -1

    # Invert Source
    cmd_src.invert(state)
    assert state.current_attack.source_instance_id == -1
