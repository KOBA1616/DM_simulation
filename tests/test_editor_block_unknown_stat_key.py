# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.consistency import validate_command_list


def test_stat_command_unknown_key_warns() -> None:
    """STAT コマンドで未知の stat キーを指定した場合、保存前に警告が返ることを検証する。"""
    cmds = [
        {"type": "STAT", "str_param": "FOO_BAR", "amount": 1}
    ]

    warnings = validate_command_list(cmds)
    assert any("FOO_BAR" in w for w in warnings), (
        "未知の統計キーを許容しています：保存前にブロックしてください"
    )
