# -*- coding: utf-8 -*-
import os
import re


def find_action_refs(root_dir: str):
    pattern = re.compile(r"\bACTION\b")
    matches = []
    for dirpath, dirs, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    text = fh.read()
            except Exception:
                continue
            for m in pattern.finditer(text):
                # Record file and surrounding context line
                # Get line number
                lineno = text[:m.start()].count('\n') + 1
                matches.append((path.replace('\\', '/'), lineno))
    return matches


def test_no_action_refs_in_editor():
    """RED: Fail if 'ACTION' token still appears in editor sources.

    This test is the RED phase for D-1: it should fail until ACTION references
    are removed or migrated. It intentionally asserts zero occurrences.
    """
    root = os.path.join('dm_toolkit', 'gui', 'editor')
    refs = find_action_refs(root)
    assert len(refs) == 0, f"Found ACTION references ({len(refs)}): {refs[:10]}"
