import ast, sys
p='dm_ai_module.py'
try:
    with open(p,'r',encoding='utf-8') as f:
        src=f.read()
    ast.parse(src)
    print('OK')
except SyntaxError as e:
    print('SYNTAX ERROR', e.msg, 'line', e.lineno)
    # Print context lines
    lines=src.splitlines()
    for i in range(max(0,e.lineno-5), min(len(lines), e.lineno+2)):
        print(f'{i+1:4}: {lines[i]}')
    sys.exit(1)
except Exception as e:
    print('ERR',e)
    sys.exit(2)
