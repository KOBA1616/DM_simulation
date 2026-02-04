import sys, os
sys.path.insert(0, os.getcwd())
try:
    import dm_ai_module
    print([a for a in dir(dm_ai_module) if 'native' in a])
except Exception as e:
    print('import error:', e)
    import traceback
    traceback.print_exc()
