# -*- coding: utf-8 -*-
import pathlib

TARGETS = [
    'dm_toolkit/gui/editor/forms/parts/filter_widget.py',
    'dm_toolkit/gui/editor/forms/parts/condition_widget.py',
    'dm_toolkit/gui/editor/forms/card_form.py',
]


def test_no_raw_connect_in_target_files():
    # file is at python/tests/unit/...; repo root is 3 parents up
    root = pathlib.Path(__file__).resolve().parents[3]
    found = []
    for rel in TARGETS:
        p = root / rel
        assert p.exists(), f"Target file missing: {p}"
        text = p.read_text(encoding='utf-8')
        if '.connect(' in text:
            found.append(str(p))
    assert not found, 'Found raw .connect calls in: ' + ', '.join(found)
