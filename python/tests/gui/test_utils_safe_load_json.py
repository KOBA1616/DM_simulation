# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import os
from dm_toolkit.gui.editor.utils import safe_load_json


def test_safe_load_json_primary_exists(tmp_path):
    p = tmp_path / "primary.json"
    p.write_text('[1,2,3]', encoding='utf-8')
    res = safe_load_json(str(p))
    assert res == [1,2,3]


def test_safe_load_json_fallback(tmp_path):
    primary = tmp_path / "nope.json"
    fb = tmp_path / "fb.json"
    fb.write_text('{"k": 5}', encoding='utf-8')
    res = safe_load_json(str(primary), [str(fb)])
    assert res == {"k": 5}


def test_safe_load_json_all_missing(tmp_path):
    res = safe_load_json(str(tmp_path / 'missing.json'), [])
    assert res is None
