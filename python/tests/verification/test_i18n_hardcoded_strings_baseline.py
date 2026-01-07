# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    # python/tests/verification/*.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[3]


def _is_probably_user_visible_english(text: str) -> bool:
    # Heuristic: only flag strings that contain ASCII letters.
    # This keeps the detector actionable while we migrate.
    return bool(re.search(r"[A-Za-z]", text))


def _extract_string_literal(arg: str) -> str | None:
    """Extract a simple single-line Python string literal content.

    Supports "..." and '...' without escaped quotes across lines.
    """
    m = re.match(r"\s*([\"\'])(.*)\1\s*$", arg)
    if not m:
        return None
    return m.group(2)


def _scan_gui_for_hardcoded_strings(root: Path) -> set[str]:
    """Return a set of stable signatures for hard-coded English-ish UI strings."""

    gui_root = root / "dm_toolkit" / "gui"

    # Match common Qt constructors / setters with a first argument string literal.
    # We only look at single-line literals to keep it robust.
    patterns = [
        re.compile(r"\bQLabel\(\s*([^,\)]+)"),
        re.compile(r"\bQPushButton\(\s*([^,\)]+)"),
        re.compile(r"\bQGroupBox\(\s*([^,\)]+)"),
        re.compile(r"\bQDockWidget\(\s*([^,\)]+)"),
        re.compile(r"\bQAction\(\s*([^,\)]+)"),
        re.compile(r"\bsetWindowTitle\(\s*([^\)]+)\)"),
        re.compile(r"\bsetText\(\s*([^\)]+)\)"),
        re.compile(r"\bsetToolTip\(\s*([^\)]+)\)"),
        re.compile(r"\bsetPlaceholderText\(\s*([^\)]+)\)"),
        re.compile(r"\bQMessageBox\.(?:information|warning|critical)\([^,]+,\s*([^,\)]+)"),
    ]

    # Ignore when tr(...) appears before the first string (e.g. QLabel(tr("..."))).
    tr_call = re.compile(r"\btr\s*\(")

    hits: set[str] = set()

    for file_path in gui_root.rglob("*.py"):
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for idx, line in enumerate(lines, start=1):
            if "tr(" in line:
                # We'll still check, but we avoid flagging if the first arg is tr(...)
                pass

            for pat in patterns:
                m = pat.search(line)
                if not m:
                    continue

                arg = m.group(1).strip()
                if arg.startswith("tr("):
                    continue

                lit = _extract_string_literal(arg)
                if lit is None:
                    continue

                if not lit:
                    continue

                if not _is_probably_user_visible_english(lit):
                    continue

                rel = file_path.relative_to(root).as_posix()
                # Signature includes location + literal to keep diffs understandable.
                hits.add(f"{rel}:{idx}:{lit}")

    return hits


def _load_baseline(root: Path) -> set[str]:
    baseline_path = root / "python" / "tests" / "verification" / "i18n_hardcoded_strings_baseline.txt"
    if not baseline_path.exists():
        return set()
    return {line.strip() for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def test_gui_hardcoded_english_strings_do_not_increase() -> None:
    """Gradual i18n enforcement: do not increase hard-coded English UI strings.

    This test is intentionally baseline-based so we can migrate screen-by-screen
    without breaking CI immediately.
    """

    root = _repo_root()
    current = _scan_gui_for_hardcoded_strings(root)
    baseline = _load_baseline(root)

    # If no baseline exists yet, fail with instructions.
    if not baseline:
        msg = [
            "Missing baseline file: python/tests/verification/i18n_hardcoded_strings_baseline.txt",
            "Generate it by running:",
            "  C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe -c \"from python.tests.verification.test_i18n_hardcoded_strings_baseline import _repo_root,_scan_gui_for_hardcoded_strings; r=_repo_root(); print('\\n'.join(sorted(_scan_gui_for_hardcoded_strings(r))))\"",
        ]
        raise AssertionError("\n".join(msg))

    added = sorted(current - baseline)
    if added:
        raise AssertionError(
            "New hard-coded English UI strings detected (add tr() or update baseline intentionally):\n"
            + "\n".join(f"- {x}" for x in added)
        )
