import dm_ai_module as dm

print(sorted([x for x in dir(dm) if not x.startswith('_')]))
