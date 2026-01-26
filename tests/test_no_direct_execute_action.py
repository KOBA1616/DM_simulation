import os
import re
import pathlib


IGNORED_PATHS = [
    "dm_ai_module.py",
    "tests/",
    "scripts/",
    "bin/",
    "dm_toolkit/gui/",
]


def is_ignored(path: str) -> bool:
    for p in IGNORED_PATHS:
        if p.endswith("/"):
            if str(path).startswith(p):
                return True
        else:
            if str(path).endswith(p):
                return True
    return False


def test_no_direct_execute_action_usage():
    """Fail if repository Python files (outside ignored paths) call `execute_action(` directly.

    This test enforces migration to the command-first APIs. It is intentionally
    conservative and can be relaxed for documented shims.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r"\bexecute_action\s*\(")
    offenders = []
    for p in repo_root.rglob("*.py"):
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        if is_ignored(rel):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if pattern.search(text):
            offenders.append(rel)

    if offenders:
        raise AssertionError("Direct execute_action(...) usage found in: " + ", ".join(sorted(offenders)))
