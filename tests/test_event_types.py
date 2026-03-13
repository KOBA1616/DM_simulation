from dm_toolkit import event_types


def test_event_types_exist():
    assert isinstance(event_types.STATE_CHANGED, str)
    assert isinstance(event_types.ACTION_EXECUTED, str)
    assert isinstance(event_types.COMMAND_EMITTED, str)
    assert isinstance(event_types.TURN_STARTED, str)
    assert isinstance(event_types.TURN_ENDED, str)


def test_event_types_values():
    assert event_types.STATE_CHANGED == 'STATE_CHANGED'
    assert event_types.ACTION_EXECUTED == 'ACTION_EXECUTED'
