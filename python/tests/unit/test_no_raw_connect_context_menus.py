# -*- coding: utf-8 -*-
import pathlib

TARGET = 'dm_toolkit/gui/editor/context_menus.py'


def test_no_raw_connect_in_context_menus():
    root = pathlib.Path(__file__).resolve().parents[3]
    p = root / TARGET
    assert p.exists(), f"Target file missing: {p}"
    text = p.read_text(encoding='utf-8')
    assert '.connect(' not in text, f'Found raw .connect in: {p}'
