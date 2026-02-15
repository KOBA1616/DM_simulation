from pathlib import Path
p = Path('CMakeLists.txt')
s = p.read_text(encoding='utf-8')
start = s.find('set(SRC_ENGINE')
if start==-1:
    print('SRC_ENGINE block not found')
    raise SystemExit(1)
open_paren = s.find('(', start)
# find matching closing ) of set block
idx = open_paren+1
paren = 1
while idx < len(s) and paren>0:
    if s[idx]=='(':
        paren+=1
    elif s[idx]==')':
        paren-=1
    idx+=1
end = idx
block = s[open_paren+1:end-1]
lines = block.splitlines()
kept = []
from os import path
for line in lines:
    lstrip = line.strip()
    if not lstrip or lstrip.startswith('#'):
        kept.append(line)
        continue
    # assume it's a source file path maybe with trailing comment
    parts = line.split('#',1)
    filepart = parts[0].strip().rstrip(',')
    filepart = filepart.strip()
    # remove trailing commas
    filepart = filepart.rstrip(',')
    if not filepart:
        kept.append(line)
        continue
    file_path = Path(filepart)
    if file_path.exists():
        kept.append(line)
    else:
        kept.append('    # MISSING: ' + line.strip())
        print('Missing:', filepart)
new_block = '(' + '\n'.join(kept) + '\n)'
new_s = s[:open_paren] + new_block + s[end:]
p.write_text(new_s, encoding='utf-8')
print('Wrote sanitized CMakeLists.txt')
