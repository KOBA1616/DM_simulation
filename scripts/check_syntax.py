import py_compile
import sys
try:
    py_compile.compile('dm_ai_module.py', doraise=True)
    print('OK')
except py_compile.PyCompileError as e:
    print('COMPILE ERROR')
    print(e)
    sys.exit(1)
