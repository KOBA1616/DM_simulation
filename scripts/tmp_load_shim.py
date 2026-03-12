import importlib.util, json
from pathlib import Path
p = Path(__file__).resolve().parent.parent / 'dm_ai_module.py'
if not p.exists():
    print('NO_SHIM')
    raise SystemExit(2)
spec = importlib.util.spec_from_file_location('dm_ai_module_for_audit', str(p))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
syms = sorted([x for x in dir(m) if not x.startswith('_')])
print(json.dumps(syms, ensure_ascii=False))
