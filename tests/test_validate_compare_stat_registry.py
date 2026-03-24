# -*- coding: utf-8 -*-
"""TDD: COMPARE_STAT の stat_key がレジストリに存在することを検証するテスト。"""

from dm_toolkit.gui.editor.validators_shared import ConditionValidator
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_compare_stat_unknown_key_is_invalid():
    cond = {"type": "COMPARE_STAT", "stat_key": "UNKNOWN_STAT", "op": ">=", "value": 1}
    errs = ConditionValidator.validate_trigger(cond)
    assert any("COMPARE_STAT" in e or "stat_key" in e for e in errs), f"期待: UNKNOWN_STAT は無効, got: {errs}"


def test_compare_stat_known_key_is_valid():
    # Pick a canonical key from CardTextResources
    key = CardTextResources.COMPARE_STAT_EDITOR_KEYS[0]
    cond = {"type": "COMPARE_STAT", "stat_key": key, "op": ">=", "value": 1}
    errs = ConditionValidator.validate_trigger(cond)
    assert errs == [], f"期待: {key} は有効, got: {errs}"
