import sys
import importlib.machinery
import types

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
