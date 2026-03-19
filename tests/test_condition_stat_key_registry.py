# -*- coding: utf-8 -*-
"""条件統計キーのレジストリ同期を検証する契約テスト。"""

from __future__ import annotations

from dm_toolkit.gui.editor.forms.parts.condition_widget import COMMON_COMPARE_STAT_KEYS
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_compare_stat_keys_are_defined_in_stat_key_map() -> None:
    """COMPARE_STAT候補キーがすべてSTAT_KEY_MAPでラベル化されること。"""
    missing_keys: list[str] = [
        key for key in CardTextResources.COMPARE_STAT_EDITOR_KEYS
        if key not in CardTextResources.STAT_KEY_MAP
    ]
    assert not missing_keys, (
        f"STAT_KEY_MAPに未定義のCOMPARE_STATキーがあります: {missing_keys}\n"
        "再発防止: 統計キー追加時は CardTextResources.COMPARE_STAT_EDITOR_KEYS と "
        "CardTextResources.STAT_KEY_MAP を同時更新すること"
    )


def test_condition_widget_and_text_resource_share_same_compare_stat_keys() -> None:
    """ConditionEditorWidgetとテキスト資源で統計キー定義が一致すること。"""
    assert COMMON_COMPARE_STAT_KEYS == list(CardTextResources.COMPARE_STAT_EDITOR_KEYS), (
        "ConditionEditorWidget の統計キー候補が CardTextResources の定義と不一致です。\n"
        "再発防止: UI側に統計キーを直書きせず、共通定義を参照すること"
    )


def test_compare_stat_keys_have_human_readable_labels() -> None:
    """統計キーラベルが生キーのフォールバックにならないこと。"""
    for key in CardTextResources.COMPARE_STAT_EDITOR_KEYS:
        label: str = CardTextResources.get_stat_key_label(key)
        assert label != key, (
            f"キー {key} の日本語ラベル未定義。\n"
            "再発防止: 追加キーにはSTAT_KEY_MAPの翻訳を必須にすること"
        )
