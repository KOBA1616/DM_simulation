from pathlib import Path
p=Path('dm_ai_module.py').read_text().splitlines()
for i,line in enumerate(p, start=1):
    if 1760 <= i <= 1800:
        print(f"{i:5}: {line.rstrip()}")
