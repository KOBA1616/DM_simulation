from pathlib import Path
p=Path('dm_ai_module.py')
src=p.read_text().splitlines()
for i in range(1008,1040):
    if i-1 < len(src):
        print(f"{i:5}: {repr(src[i-1])}")
