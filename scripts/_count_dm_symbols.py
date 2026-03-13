import sys, pathlib
sys.path.insert(0, str(pathlib.Path().absolute()))
try:
    import dm_ai_module as dm
except Exception as e:
    print('IMPORT_ERROR:'+str(e))
    sys.exit(1)
syms = sorted([x for x in dir(dm) if not x.startswith('_')])
print(len(syms))
