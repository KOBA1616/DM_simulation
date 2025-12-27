import sys, os, inspect
sys.path.insert(0, os.getcwd())
import dm_ai_module
print('PhaseManager module:', dm_ai_module.PhaseManager.__module__)
print('PhaseManager file:', inspect.getsourcefile(dm_ai_module.PhaseManager))
src = inspect.getsource(dm_ai_module.PhaseManager)
print('\n'.join(src.splitlines()[:120]))
