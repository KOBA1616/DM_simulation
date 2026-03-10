import os
import sys
import pathlib

# Ensure repository root is on sys.path so dm_ai_module.py is discoverable
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Force fallback mode for this check
os.environ['DM_DISABLE_NATIVE'] = '1'

import dm_ai_module
print(getattr(dm_ai_module,'__file__',None))
print('IS_NATIVE=', getattr(dm_ai_module,'IS_NATIVE',None))
print('IS_FALLBACK=', getattr(dm_ai_module,'IS_FALLBACK',None))
import os
os.environ['DM_DISABLE_NATIVE'] = '1'
import dm_ai_module
print(getattr(dm_ai_module,'__file__',None))
print('IS_NATIVE=', getattr(dm_ai_module,'IS_NATIVE',None))
print('IS_FALLBACK=', getattr(dm_ai_module,'IS_FALLBACK',None))
