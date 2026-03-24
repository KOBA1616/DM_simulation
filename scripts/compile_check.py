import py_compile, traceback
try:
    py_compile.compile('dm_ai_module.py', doraise=True)
    print('Compiled OK')
except Exception:
    traceback.print_exc()
