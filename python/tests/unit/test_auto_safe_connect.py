# -*- coding: utf-8 -*-
from scripts.auto_safe_connect import replace_connects_in_text, ensure_safe_connect_import


def test_replace_simple():
    src = 'btn.clicked.connect(lambda: do())\n'
    new, n = replace_connects_in_text(src)
    assert n == 1
    assert "safe_connect(btn, 'clicked', lambda: do())" in new


def test_replace_dotted():
    src = 'self.timer.timeout.connect(cb)\n'
    new, n = replace_connects_in_text(src)
    assert n == 1
    assert "safe_connect(self.timer, 'timeout', cb)" in new


def test_no_change_when_safe():
    src = 'safe_connect(btn, "clicked", handler)\n'
    new, n = replace_connects_in_text(src)
    assert n == 0
    assert new == src


def test_ensure_import_added():
    src = 'import os\n\nbtn.clicked.connect(handler)\n'
    new, n = replace_connects_in_text(src)
    assert n == 1
    updated, added = ensure_safe_connect_import(new)
    assert added is True
    assert 'from dm_toolkit.gui.editor.forms.signal_utils import safe_connect' in updated
