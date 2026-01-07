import os
import sys

# NOTE:
# sitecustomize.py is executed automatically by Python when it is importable.
# For clone-based distribution, keep behavior minimal and non-destructive.
# Avoid implicitly mutating sys.path beyond the repo root, and never delete caches.

_ROOT_DIR = os.path.dirname(__file__)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Optional dev convenience: add build/bin to import path only when explicitly enabled.
if os.environ.get('DM_SIMULATION_DEV', '').strip() == '1':
    for subdir in ['bin', 'build']:
        path = os.path.join(_ROOT_DIR, subdir)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
