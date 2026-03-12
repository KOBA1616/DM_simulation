import json
import sys
try:
    import dm_ai_module as dm
except Exception as e:
    print('IMPORT_ERROR:'+str(e))
    sys.exit(1)
syms = sorted([x for x in dir(dm) if not x.startswith('_')])
print(json.dumps(syms, ensure_ascii=False))
