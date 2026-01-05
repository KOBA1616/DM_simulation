import sys
import os

# Ensure repository root is searched first.
_ROOT_DIR = os.path.dirname(__file__)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Automatically add the 'bin/' and 'build/' directories to sys.path to help find compiled extensions
# This is a backup if dm_ai_module.py logic is bypassed or for other shared libs
for subdir in ['bin', 'build']:
    path = os.path.join(_ROOT_DIR, subdir)
    if os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)

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
