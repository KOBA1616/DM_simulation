from pathlib import Path
s=Path('dm_ai_module.py').read_text()
for i,l in enumerate(s.splitlines(),start=1):
    if 'class ' in l and 'Mana' in l:
        print(i,l)
