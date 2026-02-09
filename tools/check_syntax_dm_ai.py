import ast, sys
p = 'dm_ai_module.py'
try:
    with open(p, 'r', encoding='utf-8') as f:
        s = f.read()
    ast.parse(s, p)
    print('OK')
except SyntaxError as e:
    print('SYNTAX_ERROR', e.lineno, e.offset, e.text.strip() if e.text else None)
    sys.exit(1)
except Exception as e:
    print('ERR', e)
    sys.exit(2)
