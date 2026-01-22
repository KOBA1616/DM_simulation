p='dm_ai_module.py'
with open(p,'r',encoding='utf-8') as f:
    lines=f.readlines()
for i in range(1720,1780):
    print(f"{i+1:5d}: {len(lines[i])} bytes | {repr(lines[i])}")
