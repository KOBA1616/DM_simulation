"""scripts/ conftest.

Historically this folder had its own dm_ai_module shim.
The repository now provides a canonical loader at repo root (dm_ai_module.py),
so we avoid force-loading anything here.
"""

import os
import sys


def pytest_configure() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
