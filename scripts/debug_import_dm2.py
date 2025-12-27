import importlib, traceback
try:
    m = importlib.import_module('dm_ai_module')
    print('Imported dm_ai_module, has GameState=', hasattr(m, 'GameState'))
    keys = [k for k in dir(m) if not k.startswith('__')][:60]
    print('keys sample:', keys)
    print('module file:', getattr(m, '__file__', None))
except Exception:
    traceback.print_exc()
