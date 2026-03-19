#!/usr/bin/env python3
"""Audit utility: scan data/cards.json for STAT-like keys and report unknown ones.

Usage:
  python tools/stat_key_audit.py

Exits with code 0 when no unknown keys found, otherwise prints list and exits 2.
"""

import json
import re
import sys
from pathlib import Path
from dm_toolkit.gui.editor.text_resources import CardTextResources


def gather_pipeline_stat_names() -> set:
    p = Path("src/engine/infrastructure/pipeline/pipeline_executor.cpp")
    if not p.exists():
        return set()
    text = p.read_text(encoding="utf-8")
    names = set(re.findall(r'stat_name\s*==\s*"([A-Z0-9_]+)"', text))
    names.update(re.findall(r'else if \(stat_name == "([A-Z0-9_]+)"\)', text))
    return names


def collect_data_keys() -> set:
    p = Path("data/cards.json")
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    keys = set()

    def scan(obj):
        if isinstance(obj, dict):
            t = obj.get("type")
            if isinstance(t, str) and t.upper() == "QUERY":
                sp = obj.get("str_param")
                if isinstance(sp, str):
                    keys.add(sp)
            for v in obj.values():
                scan(v)
        elif isinstance(obj, list):
            for it in obj:
                scan(it)

    scan(data)
    return keys


def looks_like_stat(s: str) -> bool:
    return bool(re.match(r'^[A-Z0-9_]+$', s)) and ("_" in s)


def main() -> int:
    data_keys = collect_data_keys()
    pipeline_names = gather_pipeline_stat_names()
    editor_keys = set(CardTextResources.STAT_KEY_MAP.keys()) | set(CardTextResources.COMPARE_STAT_EDITOR_KEYS)

    extra_allowed = {"SELECT_TARGET"}
    allowed = pipeline_names | editor_keys | extra_allowed

    unknown = sorted([k for k in data_keys if looks_like_stat(k) and k not in allowed])

    if unknown:
        print("Unknown stat-like keys found in data/cards.json:")
        for k in unknown:
            print(" - ", k)
        print()
        print("Allowed sources: pipeline_executor.cpp stat_name literals, CardTextResources.STAT_KEY_MAP/COMPARE_STAT_EDITOR_KEYS, plus hard-coded helpers.")
        return 2

    print("No unknown stat-like keys found in data/cards.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
