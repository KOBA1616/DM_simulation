# -*- coding: utf-8 -*-
import ast
import pathlib

FILES = [
    'dm_toolkit/consts.py',
    'dm_toolkit/gui/editor/consts.py',
    'dm_toolkit/gui/editor/constants.py'
]


def _collect_constants(path):
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name.isupper():
                        names.add(name)
    return names


def test_no_duplicate_top_level_constants():
    root = pathlib.Path(__file__).resolve().parents[3]
    collected = {}
    for f in FILES:
        p = root / f
        assert p.exists(), f"Missing file: {p}"
        collected[f] = _collect_constants(p)

    # find overlaps
    overlaps = {}
    files = list(collected.keys())
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            a = files[i]
            b = files[j]
            common = collected[a] & collected[b]
            if common:
                overlaps[f"{a} <-> {b}"] = sorted(common)

    assert not overlaps, f"Found duplicate top-level constants across files: {overlaps}"
