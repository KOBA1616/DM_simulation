import ast
from pathlib import Path
p=Path('dm_ai_module.py')
s=p.read_text()
try:
    ast.parse(s)
    print('OK')
except SyntaxError as e:
    print('SyntaxError:', e)
    ln=e.lineno
    start=max(1,ln-5)
    end=ln+5
    lines=s.splitlines()
    for i in range(start, min(end, len(lines))+1):
        prefix='>' if i==ln else ' '
        print(f"{prefix} {i:4}: {lines[i-1]}")
