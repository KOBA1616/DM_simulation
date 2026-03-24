"""tests/conftest.py — テストディレクトリ用 conftest

PyQt6 未インストール時は GUI テストを自動スキップする。
再発防止: GUIテストが PyQt6 なし環境で ImportError になるのを防ぐ。
"""
import importlib.util
import sys
import pytest


def _is_pyqt6_available() -> bool:
    """Return True when PyQt6 can be imported or is already stubbed in sys.modules."""
    try:
        return importlib.util.find_spec("PyQt6") is not None
    except ValueError:
        # 再発防止: ヘッドレススタブで PyQt6.__spec__ が None の場合、
        # find_spec が ValueError を送出するため sys.modules をフォールバック判定に使う。
        return "PyQt6" in sys.modules


def pytest_collection_modifyitems(items):
    """PyQt6 未インストール時はGUIテストを自動スキップする。
    再発防止: GUIテストが PyQt6 なし環境で ImportError になるのを防ぐ。
    """
    pyqt6_available = _is_pyqt6_available()
    if not pyqt6_available:
        skip_gui = pytest.mark.skip(
            reason="PyQt6 not installed (use: pip install -e '.[gui]')"
        )
        for item in items:
            if "gui" in str(item.fspath) or "gui" in item.nodeid:
                item.add_marker(skip_gui)
