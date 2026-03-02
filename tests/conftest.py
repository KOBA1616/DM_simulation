"""tests/conftest.py — テストディレクトリ用 conftest

PyQt6 未インストール時は GUI テストを自動スキップする。
再発防止: GUIテストが PyQt6 なし環境で ImportError になるのを防ぐ。
"""
import importlib.util
import pytest


def pytest_collection_modifyitems(items):
    """PyQt6 未インストール時はGUIテストを自動スキップする。
    再発防止: GUIテストが PyQt6 なし環境で ImportError になるのを防ぐ。
    """
    pyqt6_available = importlib.util.find_spec("PyQt6") is not None
    if not pyqt6_available:
        skip_gui = pytest.mark.skip(
            reason="PyQt6 not installed (use: pip install -e '.[gui]')"
        )
        for item in items:
            if "gui" in str(item.fspath) or "gui" in item.nodeid:
                item.add_marker(skip_gui)
