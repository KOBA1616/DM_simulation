p='dm_ai_module.py'
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
tries=[]
excepts=[]
for i,l in enumerate(lines, start=1):
    s=l.strip()
    if s.startswith('try:'):
        tries.append(i)
    if s.startswith('except'):
        excepts.append(i)
print('tries',len(tries),'excepts',len(excepts))
print('first 10 tries',tries[:10])
print('first 10 excepts',excepts[:10])
# show context around first mismatch (if any)
if len(tries)!=len(excepts):
    print('Mismatch:')
    print('Last try at',tries[-1])
    for j in range(max(0,tries[-1]-5), min(len(lines), tries[-1]+5)):
        print(j+1, lines[j].rstrip())
