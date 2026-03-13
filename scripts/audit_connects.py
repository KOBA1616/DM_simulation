#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan `dm_toolkit/gui/editor/forms` for raw `.connect(` uses and `safe_connect` uses.
Print a summary report suitable for updating the TDD plan.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
FORMS_DIR = ROOT / "dm_toolkit" / "gui" / "editor" / "forms"

RE_CONNECT = re.compile(r"\.connect\(")
RE_SAFE = re.compile(r"safe_connect\(")


def scan_forms() -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for p in FORMS_DIR.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        raw = len(RE_CONNECT.findall(text))
        safe = len(RE_SAFE.findall(text))
        if raw or safe:
            results[str(p.relative_to(ROOT))] = {"raw_connect": raw, "safe_connect": safe}
    return results


def print_report(results: Dict[str, Dict[str, int]]) -> None:
    total_raw = sum(v["raw_connect"] for v in results.values())
    total_safe = sum(v["safe_connect"] for v in results.values())
    print("Connect audit report for dm_toolkit/gui/editor/forms")
    print("-------------------------------------------------")
    for f, v in sorted(results.items()):
        print(f"{f}: raw_connect={v['raw_connect']}, safe_connect={v['safe_connect']}")
    print("-------------------------------------------------")
    print(f"Total raw .connect(: {total_raw}")
    print(f"Total safe_connect(: {total_safe}")
    if total_raw == 0:
        print("Status: All connections are using safe_connect or none present.")
    else:
        print("Status: Some raw .connect calls remain. Prioritize files with highest raw counts.")


def main() -> int:
    results = scan_forms()
    print_report(results)
    return 0 if sum(v["raw_connect"] for v in results.values()) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
