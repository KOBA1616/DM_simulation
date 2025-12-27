from pathlib import Path
s=Path('dm_ai_module.py').read_text()
for i,l in enumerate(s.splitlines(),start=1):
    if 'class ActionGenerator' in l:
        print('found', i)
        break
else:
    print('not found')
