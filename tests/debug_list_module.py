import dm_ai_module as d
print('module_file=', getattr(d, '__file__', None))
print([x for x in dir(d) if 'debug' in x])
print(sorted([x for x in dir(d) if x.startswith('debug_')]))
print('IS_NATIVE' in dir(d) and getattr(d, 'IS_NATIVE', False))
