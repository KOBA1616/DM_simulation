# -*- coding: utf-8 -*-
import pathlib

def test_no_raw_connect_in_widget_factory():
    p = pathlib.Path('dm_toolkit/gui/editor/widget_factory.py')
    assert p.exists(), 'widget_factory.py not found'
    src = p.read_text(encoding='utf-8')
    assert '.connect(' not in src, 'Raw .connect( found in widget_factory.py; please use safe_connect'
