#!/usr/bin/env python3
"""Check for strings passed to tr() or present in .ui files that are missing from data/locale/ja.json

Usage: python scripts/check_missing_translations.py
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Set, List

ROOT = Path(__file__).resolve().parents[1]
LOCALE = ROOT / "data" / "locale" / "ja.json"
EXCLUDE_DIRS = {"build", "build-ninja", ".git", "__pycache__", ".cmake_deps_cache", "venv", ".venv"}


def iter_source_files(root: Path):
    for path in root.rglob("*"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix in {".py", ".ui"} and path.is_file():
            yield path


def extract_tr_strings_from_py(path: Path) -> List[str]:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return []

    try:
        tree = ast.parse(src)
    except Exception:
        return []

    results: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr

            if name in {"tr", "translate"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    results.append(first.value)
    return results


def extract_from_ui(path: Path) -> List[str]:
    results: List[str] = []
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return results

    text_attrs = {"text", "title", "placeholderText", "toolTip", "whatsThis"}
    for elem in root.iter():
        for attr in text_attrs:
            val = elem.get(attr)
            if val and val.strip():
                results.append(val.strip())
    return results


def is_probably_code_token(s: str) -> bool:
    # Filter out color hex, short tokens, format-like tokens, and pure punctuation
    if not s:
        return True
    if re.fullmatch(r"#?[0-9A-Fa-f]{3,8}", s):
        return True
    if len(s) <= 1:
        return True
    # tokens like "TRANSITION", "MOVE_CARD" are ok to translate but may be keys; keep them
    # consider strings that are purely code-like (uppercase with underscores) as keys, still check
    return False


def main() -> int:
    if not LOCALE.exists():
        print(f"Locale file not found: {LOCALE}")
        return 1

    with LOCALE.open("r", encoding="utf-8") as f:
        translations = json.load(f)

    keys: Set[str] = set(translations.keys())

    found: Set[str] = set()
    for path in iter_source_files(ROOT):
        if path.suffix == ".py":
            for s in extract_tr_strings_from_py(path):
                found.add(s)
        else:
            for s in extract_from_ui(path):
                found.add(s)

    missing = []
    for s in sorted(found):
        if s in keys:
            continue
        if is_probably_code_token(s):
            continue
        # Ignore strings that contain Japanese characters already
        if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", s):
            continue
        missing.append(s)

    if missing:
        print("Missing translations (candidates):")
        for m in missing:
            print(repr(m))
        return 2
    else:
        print("No missing translation candidates found (or all filtered).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
