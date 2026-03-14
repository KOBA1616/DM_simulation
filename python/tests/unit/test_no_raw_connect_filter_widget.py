# -*- coding: utf-8 -*-
import pathlib

def test_no_raw_connect_in_filter_widget():
    p = pathlib.Path('dm_toolkit/gui/editor/forms/filter_widget.py')
    assert p.exists(), 'filter_widget.py not found'
    src = p.read_text(encoding='utf-8')
    assert '.connect(' not in src, 'Raw .connect( found in filter_widget.py; please use safe_connect'
