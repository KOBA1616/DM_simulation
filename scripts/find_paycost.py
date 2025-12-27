from pathlib import Path
p=Path('dm_ai_module.py')
src=p.read_text()
found=False
for i,l in enumerate(src.splitlines(), start=1):
    if 'def pay_cost' in l or 'class ManaSystem' in l:
        print(i, l)
        lines=src.splitlines()
        for j in range(max(1,i-5), min(len(lines), i+5)+1):
            print(f"{j:4}: {lines[j-1]}")
        found=True
        break
if not found:
    print('not found')
