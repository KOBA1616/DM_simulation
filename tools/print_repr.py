from pathlib import Path
p=Path('dm_ai_module.py').read_text()
lines=p.splitlines()
for i in range(1778,1795):
    print(i+1, repr(lines[i]))
