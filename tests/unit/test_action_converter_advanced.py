from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_send_shield_to_grave():
    act = {'type': 'SEND_SHIELD_TO_GRAVE', 'value1': 2}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'TRANSITION'
    assert cmd['from_zone'] == 'SHIELD_ZONE'
    assert cmd['to_zone'] == 'GRAVEYARD'
    assert cmd['amount'] == 2


def test_put_creature():
    act = {'type': 'PUT_CREATURE', 'value1': 1, 'str_val': 'test_template'}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'TRANSITION'
    assert cmd['to_zone'] == 'BATTLE_ZONE'
    assert cmd['amount'] == 1
    assert cmd['str_param'] == 'test_template'


def test_cost_reference():
    act = {'type': 'COST_REFERENCE', 'str_val': 'FINISH_HYPER_ENERGY', 'value1': 0}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'MUTATE'
    assert cmd['mutation_kind'] == 'COST_REFERENCE'
    assert cmd['str_param'] == 'FINISH_HYPER_ENERGY'


def test_advanced_mutate_power():
    act = {'type': 'MUTATE', 'str_val': 'POWER_MOD', 'value1': 500}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'MUTATE'
    assert cmd['mutation_kind'] == 'POWER_MOD'
    assert cmd['amount'] == 500
