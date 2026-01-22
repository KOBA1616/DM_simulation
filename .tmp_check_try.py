from pathlib import Path
p=Path('dm_ai_module.py')
s=p.read_text()
lines=s.splitlines()
stack=[]
issues=[]
for i,l in enumerate(lines, start=1):
    stripped=l.lstrip()
    indent=len(l)-len(stripped)
    if stripped.startswith('try:'):
        stack.append((i,indent))
    if stripped.startswith('except') or stripped.startswith('finally'):
        if stack:
            stack.pop()
        else:
            issues.append(('unmatched_except',i))
# after scan
if stack:
    print('Unclosed try blocks:')
    for i,ind in stack:
        print(i, ind, lines[i-1])
else:
    print('No unclosed try blocks found')
if issues:
    print('Other issues:',issues)
