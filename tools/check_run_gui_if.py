import re
p='C:/Users/ichirou/DM_simulation/scripts/run_gui.ps1'
s=open(p,encoding='utf-8').read()
for i,l in enumerate(s.splitlines(),1):
    if re.search(r"\bif\s*\(\s*\)", l):
        print(i, repr(l))
    if re.search(r"^\s*if\s*\($", l):
        print('LINE_START_IF', i, repr(l))
    if re.search(r"\bif\s*\($", l):
        print('IF_OPEN', i, repr(l))
print('Done')
