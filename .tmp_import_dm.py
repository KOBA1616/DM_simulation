import importlib, sys
try:
    import dm_ai_module
    print('import OK')
except Exception as e:
    import traceback
    traceback.print_exc()
