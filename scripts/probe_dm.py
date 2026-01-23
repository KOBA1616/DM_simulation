import inspect
import traceback
import sys
from pathlib import Path
print('Python executable:', sys.executable)
print('sys.path preview:', sys.path[:5])
# Ensure project root is on sys.path like other scripts
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
try:
    import dm_ai_module as dm
    print('Module attrs:', [a for a in dir(dm) if not a.startswith('_')])
except Exception as e:
    print('Failed to import dm_ai_module:', type(e), e)
    import traceback; traceback.print_exc()
    sys.exit(1)
print('\nHas PhaseManager:', hasattr(dm, 'PhaseManager'))
if hasattr(dm, 'PhaseManager'):
    try:
        print('PhaseManager attrs:', [a for a in dir(dm.PhaseManager) if not a.startswith('_')])
    except Exception as e:
        print('Failed to list PhaseManager attrs:', e)
GI = getattr(dm, 'GameInstance', None)
print('\nGameInstance:', GI)
if GI is not None:
    try:
        print('GameInstance dir:', [a for a in dir(GI) if not a.startswith('_')])
    except Exception as e:
        print('Failed to list GameInstance dir:', e)
    try:
        print('Attempt constructing with (0, None)')
        GI(0, None)
        print('Constructed with (0, None)')
    except Exception as e:
        print('Construct failed:')
        traceback.print_exc()
