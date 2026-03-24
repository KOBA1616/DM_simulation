# -*- coding: utf-8 -*-
"""query_mode と stat_key 共通レジストリの同期契約テスト。"""

from dm_toolkit.consts import QUERY_MODES
from dm_toolkit.stat_keys import QUERY_STAT_KEYS
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_query_modes_are_derived_from_shared_registry() -> None:
    """再発防止: QUERY_MODES は shared registry から導出されること。"""
    assert QUERY_MODES[0] == "CARDS_MATCHING_FILTER"
    assert QUERY_MODES[1:] == list(QUERY_STAT_KEYS)


def test_query_stat_keys_are_labelled() -> None:
    """再発防止: query 可能な統計キーは表示ラベルを持つこと。"""
    missing = [k for k in QUERY_STAT_KEYS if k not in CardTextResources.STAT_KEY_MAP]
    assert not missing, f"query用stat_keyに未翻訳キーがあります: {missing}"
