import ast
import sys
fn='dm_ai_module.py'
try:
    with open(fn,'r', encoding='utf-8') as f:
        src=f.read()
    ast.parse(src)
    print('OK')
except SyntaxError as e:
    print('SyntaxError', e)
    print('Line:', e.lineno)
    # print surrounding lines
    lines=src.splitlines()
    for i in range(max(0,e.lineno-3), min(len(lines), e.lineno+2)):
        print(f"{i+1}: {lines[i]!r}")
    sys.exit(1)
except Exception as e:
    print('ERR', e)
    sys.exit(2)
