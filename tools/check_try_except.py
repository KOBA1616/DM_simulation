from pathlib import Path
p=Path('dm_ai_module.py').read_text().splitlines()
for i,line in enumerate(p, start=1):
    if 'try:' in line:
        print(i, line.strip())
        for j in range(i, min(i+8, len(p)+1)):
            print('   ', j, p[j-1].rstrip())
        print()
