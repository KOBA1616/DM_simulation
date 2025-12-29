# conftest.py - ensure repository-local dm_ai_module.py is used during pytest
import sys
import os
import importlib.machinery
import importlib.util

ROOT = os.path.dirname(__file__)
shim_path = os.path.join(ROOT, 'dm_ai_module.py')

if os.path.exists(shim_path):
    try:
        loader = importlib.machinery.SourceFileLoader('dm_ai_module', shim_path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        sys.modules['dm_ai_module'] = module
    except Exception as e:
        print(f"[conftest] failed to load dm_ai_module shim: {e}")
