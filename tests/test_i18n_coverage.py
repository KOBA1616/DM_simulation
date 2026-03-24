import json
import re
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
LOCALE = ROOT / "data" / "locale" / "ja.json"
EXCLUDE_DIRS = {"build", "build-ninja", ".git", "__pycache__", ".cmake_deps_cache", "venv", ".venv"}


def iter_source_files(root: Path):
    for path in root.rglob("*"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix in {".py", ".ui"} and path.is_file():
            yield path


def extract_tr_strings_from_py(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []
    results = []
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


def extract_from_ui(path: Path):
    results = []
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
    if not s:
        return True
    if re.fullmatch(r"#?[0-9A-Fa-f]{3,8}", s):
        return True
    if len(s) <= 1:
        return True
    return False


def test_all_ui_strings_have_ja_translation():
    assert LOCALE.exists(), f"Locale file not found: {LOCALE}"
    translations = json.loads(LOCALE.read_text(encoding="utf-8"))
    keys = set(translations.keys())

    found = set()
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
        if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", s):
            continue
        missing.append(s)

    if missing:
        missing_preview = "\n".join(missing[:50])
        raise AssertionError(f"Missing translations (candidates):\n{missing_preview}\n...\nAdd them to {LOCALE}")
