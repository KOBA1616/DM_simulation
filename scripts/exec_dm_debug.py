import importlib.util, traceback, sys, os
path = os.path.abspath(os.path.join(os.getcwd(),'dm_ai_module.py'))
print('loading from', path)
try:
    spec = importlib.util.spec_from_file_location('test_dm', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print('loaded module, names sample:', [n for n in dir(m) if 'Action' in n][:100])
    print('has ActionGenerator=', hasattr(m,'ActionGenerator'))
except Exception:
    print('EXCEPTION DURING EXEC')
    traceback.print_exc()
