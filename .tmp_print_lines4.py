p='dm_ai_module.py'
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
for i in range(1756,1778):
    print(f"{i+1:5d}: {lines[i].rstrip()}")
