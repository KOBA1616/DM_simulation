import importlib, sys
try:
    m = importlib.import_module('dm_ai_module')
    print('dm_ai_module loaded from:', getattr(m, '__file__', 'builtin or extension without __file__'))
except Exception as e:
    print('import error:', e)
print('--- sys.path ---')
for p in sys.path:
    print(p)
