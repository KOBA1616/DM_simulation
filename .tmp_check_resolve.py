import dm_ai_module
print('resolve_action attr:', getattr(dm_ai_module.GenericCardSystem, 'resolve_action'))
print('callable:', callable(getattr(dm_ai_module.GenericCardSystem, 'resolve_action', None)))
try:
    import inspect
    src = inspect.getsource(dm_ai_module.GenericCardSystem.resolve_action)
    print('source snippet:\n', '\n'.join(src.splitlines()[:30]))
except Exception as e:
    print('could not get source:', e)
