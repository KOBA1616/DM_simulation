# -*- coding: utf-8 -*-
"""Contract test ensuring CompareStatEvaluator supports editor keys.

This test verifies that every key exposed in
`CardTextResources.COMPARE_STAT_EDITOR_KEYS` is recognized by the
engine's `CompareStatEvaluator` (i.e. the key string appears in
`condition_system.cpp`).
"""

from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_compare_stat_keys_present_in_engine():
    p = Path("src/engine/systems/rules/condition_system.cpp")
    src = p.read_text(encoding="utf-8")
    missing = []
    for key in CardTextResources.COMPARE_STAT_EDITOR_KEYS:
        if key not in src and key not in CardTextResources.STAT_KEY_MAP:
            missing.append(key)

    assert not missing, f"CompareStatEvaluator が未対応のキーがあります: {missing}"
