"""scripts/ conftest.

再発防止: dm_ai_module.py (Python フォールバック) は削除済み。
現在は dm_ai_module.cp312-win_amd64.pyd (ネイティブ C++) が唯一の実装になる。
"""

import os
import sys


def pytest_configure() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
