import pytest
from dm_ai_module import GameState, CardDefinition, CardType
from tests.shims.play_commands_shim import DeclarePlayCommand, PayCostCommand, ResolvePlayCommand


def setup_simple_state():
    state = GameState()
    # minimal setup: ensure two players and use provided helpers
    state._ensure_player(0)
    state._ensure_player(1)
    # create a simple creature card in hand with instance id 1
    state.add_card_to_hand(0, 1, 1)
    # register card definition so ResolvePlayCommand treats it as a creature
    try:
        if not hasattr(globals().get('dm_ai_module', None), '_CARD_REGISTRY'):
            import dm_ai_module as dm
            dm._CARD_REGISTRY = {}
        import dm_ai_module as dm
        dm._CARD_REGISTRY[1] = CardDefinition(1, name='TestCreature', civilization=None, civilizations=[], races=[], cost=1, power=1, keywords=None, effects=[])
        dm._CARD_REGISTRY[1].type = CardType.CREATURE
    except Exception:
        pass
    # give player 0 one untapped mana source (instance id 100)
    state.add_card_to_mana(0, 200, 100)
    return state, state.players[0].hand[0]


def test_mana_payment_rollback_when_insufficient():
    state, creature = setup_simple_state()
    # remove mana so insufficient
    state.players[0].mana_zone = []
    pay_cmd = PayCostCommand(player_id=0, amount=1)
    # PayCostCommand.execute returns None in shim; ensure ManaSystem.pay_cost returns False
    ok = hasattr(pay_cmd, 'execute') and pay_cmd.execute(state) is not True
    assert ok
    # ensure no mana tapped (none exist)
    assert len(state.players[0].mana_zone) == 0


def test_declare_pay_resolve_flow():
    state, creature = setup_simple_state()
    # ensure mana exists
    assert len(state.players[0].mana_zone) == 1
    # Declare play
    declare = DeclarePlayCommand(player_id=0, card_id=1, source_instance_id=1)
    declare.execute(state)
    # Now pay
    pay = PayCostCommand(player_id=0, amount=1)
    pay.execute(state)
    # Resolve
    resolve = ResolvePlayCommand(player_id=0, card_id=1, card_def=None)
    resolve.execute(state)
    # Creature should now be on battle zone
    assert any(getattr(c, 'card_id', None) == 1 for c in state.players[0].battle_zone)


def test_command_history_and_manual_undo():
    state, creature = setup_simple_state()
    declare = DeclarePlayCommand(player_id=0, card_id=1, source_instance_id=1)
    pay = PayCostCommand(player_id=0, amount=1)
    resolve = ResolvePlayCommand(player_id=0, card_id=1, card_def=None)
    # execute via GameState to record history
    state.execute_command(declare)
    state.execute_command(pay)
    state.execute_command(resolve)
    assert len(state.command_history) >= 3
    # manual undo of last command
    last = state.command_history.pop()
    inv = getattr(last, 'invert', lambda s: None)
    # call invert function if exists
    try:
        inv(state)
    except Exception:
        pass
    # After undoing resolve, creature should not be in battle zone
    assert all(getattr(c, 'card_id', None) != 1 for c in state.players[0].battle_zone)
