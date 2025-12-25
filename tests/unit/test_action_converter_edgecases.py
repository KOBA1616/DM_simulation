from dm_toolkit.gui.editor.action_converter import ActionConverter


def assert_cmd_valid_or_legacy(cmd):
    assert isinstance(cmd, dict)
    assert 'type' in cmd
    assert 'uid' in cmd


def test_move_card_to_transition():
    act = {"type": "MOVE_CARD", "destination_zone": "BATTLE_ZONE", "source_zone": "HAND"}
    cmd = ActionConverter.convert(act)
    assert_cmd_valid_or_legacy(cmd)


def test_draw_card():
    act = {"type": "DRAW_CARD", "value1": 2, "scope": "PLAYER_SELF"}
    cmd = ActionConverter.convert(act)
    assert_cmd_valid_or_legacy(cmd)
    assert cmd.get('type') == 'DRAW_CARD'
    assert cmd.get('amount') == 2


def test_select_target_becomes_query():
    act = {"type": "SELECT_TARGET", "filter": {"zones": ["BATTLE_ZONE"], "count": 1}}
    cmd = ActionConverter.convert(act)
    assert_cmd_valid_or_legacy(cmd)
    assert cmd.get('type') == 'QUERY'


def test_apply_modifier_cost():
    act = {"type": "APPLY_MODIFIER", "str_val": "COST", "value1": -1, "scope": "TARGET_SELF"}
    cmd = ActionConverter.convert(act)
    assert_cmd_valid_or_legacy(cmd)
    assert cmd.get('type') == 'MUTATE'


def test_grant_keyword():
    act = {"type": "GRANT_KEYWORD", "str_val": "BLOCKER", "value1": 1}
    cmd = ActionConverter.convert(act)
    assert_cmd_valid_or_legacy(cmd)
    assert cmd.get('type') == 'ADD_KEYWORD' or cmd.get('legacy_warning')


def test_mekraids_flagged_legacy():
    act = {"type": "MEKRAID"}
    cmd = ActionConverter.convert(act)
    assert isinstance(cmd, dict)
    assert cmd.get('legacy_warning', False) is True
