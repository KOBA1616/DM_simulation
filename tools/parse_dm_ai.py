import ast
import sys
p='dm_ai_module.py'
with open(p,'r',encoding='utf-8') as f:
    s=f.read()
try:
    ast.parse(s)
    print('PARSE_OK')
except SyntaxError as e:
    print('SYNTAX', e, 'at', e.lineno, e.offset)
    sys.exit(2)
