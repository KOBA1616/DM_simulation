# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.consistency import validate_command_list


def test_query_nested_params_str_param_is_accepted() -> None:
    """QUERY の mode が params.str_param にある場合は未設定警告にならない。"""
    cmds = [
        {
            "type": "QUERY",
            "params": {
                "str_param": "CARDS_MATCHING_FILTER",
                "target_filter": {"zones": ["BATTLE_ZONE"]},
            },
        }
    ]

    warnings = validate_command_list(cmds)
    assert not any("Query Mode" in w for w in warnings)


def test_query_nested_params_query_mode_legacy_is_accepted() -> None:
    """再発防止: 旧キー params.query_mode でも QUERY mode として解釈できること。"""
    cmds = [
        {
            "type": "QUERY",
            "params": {
                "query_mode": "CARDS_MATCHING_FILTER",
                "target_filter": {"zones": ["BATTLE_ZONE"]},
            },
        }
    ]

    warnings = validate_command_list(cmds)
    assert not any("Query Mode" in w for w in warnings)


def test_query_missing_mode_still_warns() -> None:
    """互換対応後も mode 未設定は警告されること。"""
    cmds = [{"type": "QUERY", "params": {}}]

    warnings = validate_command_list(cmds)
    assert any("Query Mode" in w for w in warnings)


def test_query_legacy_query_string_is_accepted_top_level() -> None:
    """再発防止: 旧キー query_string（トップレベル）でも QUERY mode として解釈できること。"""
    cmds = [{"type": "QUERY", "query_string": "SELECT_TARGET"}]

    warnings = validate_command_list(cmds)
    assert not any("Query Mode" in w for w in warnings)


def test_query_legacy_query_string_is_accepted_in_params() -> None:
    """再発防止: 旧キー params.query_string でも QUERY mode として解釈できること。"""
    cmds = [{"type": "QUERY", "params": {"query_string": "SELECT_TARGET"}}]

    warnings = validate_command_list(cmds)
    assert not any("Query Mode" in w for w in warnings)
