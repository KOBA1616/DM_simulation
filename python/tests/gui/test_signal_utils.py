# -*- coding: utf-8 -*-
from __future__ import annotations

from dm_toolkit.gui.editor.forms.signal_utils import safe_connect


class _Signal:
    def __init__(self) -> None:
        self.connected = []

    def connect(self, slot):
        self.connected.append(slot)


class _Widget:
    def __init__(self) -> None:
        self.textChanged = _Signal()


def test_safe_connect_returns_true_when_signal_exists() -> None:
    widget = _Widget()

    def _slot() -> None:
        return None

    result = safe_connect(widget, "textChanged", _slot)

    assert result is True
    assert widget.textChanged.connected == [_slot]


def test_safe_connect_returns_false_when_signal_missing() -> None:
    widget = _Widget()

    result = safe_connect(widget, "editingFinished", lambda: None)

    assert result is False


def test_safe_connect_returns_false_when_connect_not_callable() -> None:
    class _BadSignal:
        def __init__(self) -> None:
            self.connect = None

    class _BadWidget:
        def __init__(self) -> None:
            self.badSignal = _BadSignal()

    widget = _BadWidget()

    result = safe_connect(widget, "badSignal", lambda: None)

    assert result is False


def test_safe_connect_returns_false_when_widget_is_none() -> None:
    result = safe_connect(None, "textChanged", lambda: None)

    assert result is False
