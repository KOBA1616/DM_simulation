import dm_ai_module as dm
print('module file:', getattr(dm,'__file__',None))
print('has GameState:', hasattr(dm,'GameState'))
print('has CardDefinition:', hasattr(dm,'CardDefinition'))
print('sample keys:', [k for k in dir(dm) if not k.startswith('__')][:80])
