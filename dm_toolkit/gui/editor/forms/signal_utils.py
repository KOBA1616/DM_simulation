# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable


def safe_connect(widget: Any, signal_name: str, slot: Callable[..., Any]) -> bool:
    """Safely connect a widget signal if it exists.

    Returns:
        True when connection succeeded, False otherwise.
    """
    if widget is None or not signal_name:
        return False

    signal: Any = getattr(widget, signal_name, None)
    if signal is None:
        return False

    connect_fn: Any = getattr(signal, "connect", None)
    if not callable(connect_fn):
        return False

    try:
        connect_fn(slot)
        return True
    except Exception:
        # 再発防止: ヘッドレススタブの簡易シグナルで connect が例外化しても
        # フォーム初期化を継続し、編集UI自体が壊れないようにする。
        return False
