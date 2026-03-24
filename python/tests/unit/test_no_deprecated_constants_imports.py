# -*- coding: utf-8 -*-
import pathlib

DEP_FILE = 'dm_toolkit/gui/editor/constants.py'
BAD_IMPORT = 'dm_toolkit.gui.editor.constants'


def test_no_deprecated_constants_imports():
    root = pathlib.Path(__file__).resolve().parents[3]
    bad_uses = []
    for p in root.rglob('*.py'):
        # skip the deprecated file itself
        rel = p.relative_to(root).as_posix()
        # skip test files and the test itself to avoid self-match
        if rel.startswith('python/tests/'): 
            continue
        if rel == DEP_FILE:
            continue
        text = p.read_text(encoding='utf-8')
        if BAD_IMPORT in text:
            bad_uses.append(rel)

    assert not bad_uses, f'Found imports of deprecated constants in: {bad_uses}'
