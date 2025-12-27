import importlib
m = importlib.import_module('dm_ai_module')
print('module:', getattr(m, '__file__', None))
print('has CardRegistry:', 'CardRegistry' in dir(m))
print('names with Card:', [n for n in dir(m) if 'Card' in n])
print('sample_dir_len:', len(dir(m)))
