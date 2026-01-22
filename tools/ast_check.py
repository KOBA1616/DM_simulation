import ast
p='dm_ai_module.py'
try:
    with open(p,'r', encoding='utf-8') as f:
        s=f.read()
    ast.parse(s)
    print('OK')
except SyntaxError as e:
    print('SyntaxError', e.lineno, e.msg)
except Exception as e:
    print('Error', e)
