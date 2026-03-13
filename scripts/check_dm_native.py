import dm_ai_module as dm
print('IS_NATIVE=', getattr(dm, 'IS_NATIVE', None))
attrs = [x for x in dir(dm) if not x.startswith('_')]
print('SAMPLE attrs count=', len(attrs))
print('First 30:', attrs[:30])
