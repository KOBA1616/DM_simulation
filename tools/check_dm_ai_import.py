import traceback
try:
    import dm_ai_module
    print('OK', hasattr(dm_ai_module, 'GameInstance'))
    print('attrs:', [k for k in ('GameInstance', 'CommandSystem', 'CardStub') if hasattr(dm_ai_module, k)])
except Exception:
    traceback.print_exc()
