import sys, os
# Put repo root first
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
print('sys.path[0]=', sys.path[0])
try:
    import dm_ai_module
    print('dm_ai_module file:', getattr(dm_ai_module, '__file__', None))
    print('has LethalSolver:', hasattr(dm_ai_module, 'LethalSolver'))
    print('names sample:', [k for k in dir(dm_ai_module) if 'Lethal' in k or 'POMDP' in k or 'Neural' in k][:50])
except Exception as e:
    print('import failed:', e)
