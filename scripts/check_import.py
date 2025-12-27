import sys, os, traceback
print('cwd=', os.getcwd())
print('sys.path[0]=', sys.path[0])
print('len(sys.path)=', len(sys.path))
try:
    import dm_ai_module
    print('imported dm_ai_module, file=', getattr(dm_ai_module,'__file__',None))
    print('has ActionGenerator=', hasattr(dm_ai_module,'ActionGenerator'))
    print('available names sample:', [n for n in dir(dm_ai_module) if 'Action' in n][:50])
except Exception as e:
    print('IMPORT FAILED')
    traceback.print_exc()
