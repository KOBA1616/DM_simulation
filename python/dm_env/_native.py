# python/dm_env/_native.py
"""dm_ai_module (C++拡張) の唯一のロードポイント。

【禁止事項】
- このファイルに PyQt6 / PySide6 の import を追加しないこと。
- dm_env 配下の全ファイルで PyQt6 import を禁止する。
- 再発防止: GUI依存がヘッドレス層に混入すると CI・Docker 環境が壊れる。
"""
from __future__ import annotations
import importlib
from typing import Any

_module: Any = None


def get_module() -> Any:
    """dm_ai_module を遅延ロードして返す。"""
    global _module
    if _module is None:
        try:
            _module = importlib.import_module("dm_ai_module")
        except ImportError as e:
            raise RuntimeError(
                "dm_ai_module (C++拡張) のロードに失敗しました。\n"
                "ビルドを確認してください: .\\scripts\\build.ps1\n"
                f"原因: {e}"
            ) from e
    return _module
