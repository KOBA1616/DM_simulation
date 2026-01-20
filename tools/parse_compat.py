import ast, sys
p = r"C:\Users\ichirou\DM_simulation\dm_toolkit\engine\compat.py"
try:
    s = open(p, 'r', encoding='utf-8').read()
    ast.parse(s)
    print('OK')
except Exception as e:
    print('ERROR:', type(e).__name__, e)
    try:
        lineno = e.lineno
        offset = e.offset
        print('lineno', lineno, 'offset', offset)
        for i, L in enumerate(s.splitlines(), start=1):
            if i >= lineno-3 and i <= lineno+3:
                print(f"{i:04d}: {L}")
    except Exception:
        pass
    sys.exit(1)
