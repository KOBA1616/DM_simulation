from pathlib import Path
p=Path('dm_ai_module.py')
s= p.read_text()
for i,l in enumerate(s.splitlines(),start=1):
    if 'ManaSystem' in l:
        print(i, l)
