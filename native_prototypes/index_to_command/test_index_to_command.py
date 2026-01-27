from native_prototypes.index_to_command.index_to_command import index_to_command


def test_index_to_command_basic():
    # PASS
    assert index_to_command(0)['type'] == 'PASS'
    # MANA_CHARGE region
    for i in (1, 5, 19):
        d = index_to_command(i)
        assert d['type'] == 'MANA_CHARGE'
        assert d['slot_index'] == i
    # PLAY_FROM_ZONE region
    for i in (20, 25, 45):
        d = index_to_command(i)
        assert d['type'] == 'PLAY_FROM_ZONE'
        assert d['slot_index'] == i - 20
