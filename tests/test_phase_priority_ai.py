import dm_ai_module as dm


def _make_cmd(t: dm.CommandType) -> dm.CommandDef:
    c = dm.CommandDef()
    c.type = t
    c.owner_id = 0
    return c


def test_mana_phase_prefers_mana_charge():
    gs = dm.GameState(0)
    gs.current_phase = dm.Phase.MANA

    actions = [
        _make_cmd(dm.CommandType.PASS),
        _make_cmd(dm.CommandType.MANA_CHARGE),
    ]

    ai = dm.SimpleAI()
    idx = ai.select_action(actions, gs)
    assert actions[idx].type == dm.CommandType.MANA_CHARGE


def test_attack_phase_prefers_attack():
    gs = dm.GameState(0)
    gs.current_phase = dm.Phase.ATTACK

    actions = [
        _make_cmd(dm.CommandType.PASS),
        _make_cmd(dm.CommandType.ATTACK_PLAYER),
    ]

    ai = dm.SimpleAI()
    idx = ai.select_action(actions, gs)
    assert actions[idx].type in (dm.CommandType.ATTACK_PLAYER, dm.CommandType.ATTACK_CREATURE)


def test_block_phase_prefers_block():
    gs = dm.GameState(0)
    gs.current_phase = dm.Phase.BLOCK

    actions = [
        _make_cmd(dm.CommandType.PASS),
        _make_cmd(dm.CommandType.BLOCK),
    ]

    ai = dm.SimpleAI()
    idx = ai.select_action(actions, gs)
    assert actions[idx].type == dm.CommandType.BLOCK
