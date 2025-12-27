import dm_ai_module
print('module_file=' + str(getattr(dm_ai_module,'__file__', 'built-in')))
print('has_LethalSolver=' + str(hasattr(dm_ai_module,'LethalSolver')))
print('names_sample=' + ','.join([k for k in dir(dm_ai_module) if 'Lethal' in k or 'POMDP' in k or 'Neural' in k][:50]))
