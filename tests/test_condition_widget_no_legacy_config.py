# -*- coding: utf-8 -*-
"""
RED test: detect legacy `CONDITION_UI_CONFIG` presence in the
`condition_widget.py` source. The goal is to require usage of the
declarative `schema_config.get_condition_form_fields` and
`CardTextResources` as the single source of truth instead of a
duplicated `CONDITION_UI_CONFIG` dictionary.
"""
from pathlib import Path


def test_no_legacy_condition_ui_config_in_source():
    p = Path("dm_toolkit/gui/editor/forms/parts/condition_widget.py")
    src = p.read_text(encoding="utf-8")
    # RED: currently the legacy dict exists; test will fail until refactor
    assert "CONDITION_UI_CONFIG" not in src, (
        "ソースに `CONDITION_UI_CONFIG` が残っています。"
        "ConditionEditorWidget は `schema_config` / `CardTextResources` の単一定義を使うようリファクタしてください。"
    )
