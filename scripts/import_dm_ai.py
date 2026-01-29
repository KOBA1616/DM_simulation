import traceback

print('Importing dm_ai_module')
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bin', 'Release')))
try:
    import dm_ai_module
    print('dm_ai_module imported successfully')
except Exception:
    print('Exception during import:')
    traceback.print_exc()
except BaseException:
    print('BaseException during import (non-Exception)')
    raise
