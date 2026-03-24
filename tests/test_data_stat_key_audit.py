# -*- coding: utf-8 -*-
"""Audit test: ensure statistic-like keys used in data/cards.json are recognized.

This test collects `str_param` and `str_val` values from card commands/conditions
and ensures any token that looks like an uppercase STAT key is present in the
engine/editor known-key sets (CardTextResources.STAT_KEY_MAP or pipeline stat
names discovered from `pipeline_executor.cpp`).
"""

import json
import re
from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def gather_pipeline_stat_names() -> set:
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    text = p.read_text(encoding="utf-8")
    names = set(re.findall(r'stat_name\s*==\s*"([A-Z0-9_]+)"', text))
    # Also capture literals used in earlier if branches
    names.update(re.findall(r'else if \(stat_name == "([A-Z0-9_]+)"\)', text))
    return names


def collect_data_keys() -> set:
    p = Path("data/cards.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    keys = set()

    def scan(obj):
        if isinstance(obj, dict):
            # Only collect str_param from QUERY-like command objects where it denotes a stat/query key
            t = obj.get("type")
            if isinstance(t, str) and t.upper() == "QUERY":
                sp = obj.get("str_param")
                if isinstance(sp, str):
                    keys.add(sp)
            # Recurse into children
            for v in obj.values():
                scan(v)
        elif isinstance(obj, list):
            for it in obj:
                scan(it)

    scan(data)
    return keys


def looks_like_stat(s: str) -> bool:
    # Heuristic: all-uppercase with underscores and at least one underscore
    return bool(re.match(r'^[A-Z0-9_]+$', s)) and ("_" in s)


def test_data_stat_keys_are_known():
    data_keys = collect_data_keys()
    pipeline_names = gather_pipeline_stat_names()
    editor_keys = set(CardTextResources.STAT_KEY_MAP.keys()) | set(CardTextResources.COMPARE_STAT_EDITOR_KEYS)

    # Some data-driven QUERY types are non-stat helpers (e.g. SELECT_TARGET)
    extra_allowed = {"SELECT_TARGET"}
    allowed = pipeline_names | editor_keys | extra_allowed

    unknown = sorted([k for k in data_keys if looks_like_stat(k) and k not in allowed])

    assert not unknown, f"data/cards.json に未登録の統計キー候補が見つかりました: {unknown}"
