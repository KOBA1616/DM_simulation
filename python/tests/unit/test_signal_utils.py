# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect


class DummySignal:
    def __init__(self):
        self.called = False
        self.slot = None

    def connect(self, slot):
        self.called = True
        self.slot = slot


class BadSignal:
    def connect(self, slot):
        raise RuntimeError("failed")


class DummyWidget:
    def __init__(self, sig=None):
        if sig is not None:
            self.my_signal = sig


def test_safe_connect_success():
    s = DummySignal()
    w = DummyWidget(s)
    ok = safe_connect(w, 'my_signal', lambda: 1)
    assert ok is True
    assert s.called is True


def test_safe_connect_missing_signal():
    w = DummyWidget()
    ok = safe_connect(w, 'no_signal', lambda: 1)
    assert ok is False


def test_safe_connect_connect_raises():
    w = DummyWidget(BadSignal())
    ok = safe_connect(w, 'my_signal', lambda: 1)
    assert ok is False
