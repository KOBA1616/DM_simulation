from pathlib import Path
s=Path('dm_ai_module.py').read_text()
for i,l in enumerate(s.splitlines(),start=1):
    if 'def setup_scenario' in l or 'setup_scenario(' in l:
        print(i,l)
    if 'class PhaseManager' in l:
        print('PhaseManager at', i)
        # print a bit after
        lines=s.splitlines()
        for j in range(i, i+80):
            if j-1 < len(lines):
                print(j, lines[j-1])
        break
