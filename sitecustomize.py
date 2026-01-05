import sys
import os

# POLICY: SOURCE LOADER PRIORITY
# ------------------------------
# We prioritize the local repository root (source) over installed packages.
# This ensures that `import dm_ai_module` (which maps to dm_ai_module.py) is
# picked up from the source tree rather than site-packages.
#
# This file is automatically executed by Python when the site module is imported.

_ROOT_DIR = os.path.dirname(__file__)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Remove stale __pycache__ directories to avoid import-file-mismatch during test collection
import shutil
try:
    for root, dirs, files in os.walk(_ROOT_DIR):
        for d in list(dirs):
            if d == '__pycache__':
                path = os.path.join(root, d)
                try:
                    shutil.rmtree(path)
                except Exception:
                    pass
except Exception:
    pass
