# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    # python/tests/verification/test_i18n_policy.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[3]


def _scan_for_forbidden_tr_fstrings(root: Path) -> list[tuple[Path, int, str]]:
    gui_root = root / "dm_toolkit" / "gui"
    pattern = re.compile(r"\btr\s*\(\s*f[\"\']")

    hits: list[tuple[Path, int, str]] = []
    for file_path in gui_root.rglob("*.py"):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # As a policy test, we only enforce for UTF-8 readable files.
            # GUI files should be UTF-8.
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append((file_path, line_no, line.strip()))

    return hits


def test_no_tr_fstrings_in_gui() -> None:
    """Policy: Do not pass f-strings into tr(). Use placeholders + .format()."""

    root = _repo_root()
    hits = _scan_for_forbidden_tr_fstrings(root)
    assert not hits, "Found forbidden tr(f\"...\") usage:\n" + "\n".join(
        f"- {p.relative_to(root)}:{line_no}: {line}" for p, line_no, line in hits
    )
