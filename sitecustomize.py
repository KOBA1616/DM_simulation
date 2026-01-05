import sys
import importlib.machinery
import types
import os
import importlib

# Ensure repository root and C++ extension output under ./bin are importable.
_ROOT_DIR = os.path.dirname(__file__)
_BIN_DIR = os.path.join(_ROOT_DIR, 'bin')
if os.path.isdir(_BIN_DIR) and _BIN_DIR not in sys.path:
    sys.path.append(_BIN_DIR)

# Ensure repository root is searched first.
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# If dm_ai_module was already imported as a namespace package (e.g. ./dm_ai_module/ exists
# without an __init__.py), reload it so the compiled extension in ./bin takes precedence.
_maybe_dm = sys.modules.get('dm_ai_module')
if _maybe_dm is not None and getattr(_maybe_dm, '__file__', None) is None:
    try:
        del sys.modules['dm_ai_module']
        importlib.import_module('dm_ai_module')
    except Exception:
        # Leave as-is; downstream will raise a clearer error if needed.
        pass

# Ensure a safe PyQt6 module entry exists with a ModuleSpec so tests using importlib.util.find_spec
# won't raise ValueError when a parent package was injected into sys.modules without a spec.
try:
    if importlib.util.find_spec('PyQt6') is None:
        mod = sys.modules.get('PyQt6')
        if mod is None:
            m = types.ModuleType('PyQt6')
            m.__spec__ = importlib.machinery.ModuleSpec('PyQt6', None)
            sys.modules['PyQt6'] = m
        else:
            if getattr(mod, '__spec__', None) is None:
                mod.__spec__ = importlib.machinery.ModuleSpec('PyQt6', None)
except Exception:
    pass

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
