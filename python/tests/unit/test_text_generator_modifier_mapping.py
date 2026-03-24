from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_format_modifier_cost_and_power():
    # COST_MODIFIER
    mod_cost = {
        "type": "COST_MODIFIER",
        "value": 2,
        "filter": {"types": ["CREATURE"]},
    }
    res_cost = CardTextGenerator._format_modifier(mod_cost)
    assert "コストを2軽減" in res_cost or "コストを2" in res_cost

    # POWER_MODIFIER
    mod_power = {
        "type": "POWER_MODIFIER",
        "value": 3,
        "filter": {"types": ["CREATURE"]},
    }
    res_power = CardTextGenerator._format_modifier(mod_power)
    assert "パワー" in res_power and ("3" in res_power or "+3" in res_power)


def test_format_modifier_grant_keyword_and_set():
    # GRANT_KEYWORD
    mod_grant = {
        "type": "GRANT_KEYWORD",
        "mutation_kind": "BLOCKER",
        "value": 1,
        "filter": {"types": ["CREATURE"]},
    }
    res_grant = CardTextGenerator._format_modifier(mod_grant)
    assert "与える" in res_grant or "与える。" in res_grant
    assert "ブロッカー" in res_grant

    # SET_KEYWORD
    mod_set = {
        "type": "SET_KEYWORD",
        "str_val": "SPEED_ATTACK",
        "filter": {"types": ["CREATURE"]},
    }
    res_set = CardTextGenerator._format_modifier(mod_set)
    assert ("得る" in res_set) or ("能力" in res_set)


def test_format_modifier_grant_keyword_uses_all_targets_when_value_zero():
    mod_grant = {
        "type": "GRANT_KEYWORD",
        "mutation_kind": "BLOCKER",
        "value": 0,
        "filter": {"types": ["CREATURE"]},
    }
    res_grant = CardTextGenerator._format_modifier(mod_grant)
    assert "選び" not in res_grant
    assert "ブロッカー" in res_grant


def test_format_modifier_grant_keyword_uses_selection_when_value_positive():
    mod_grant = {
        "type": "GRANT_KEYWORD",
        "mutation_kind": "BLOCKER",
        "value": 2,
        "filter": {"types": ["CREATURE"]},
    }
    res_grant = CardTextGenerator._format_modifier(mod_grant)
    assert "2体選び" in res_grant
    assert "ブロッカー" in res_grant
