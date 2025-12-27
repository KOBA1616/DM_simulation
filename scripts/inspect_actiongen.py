import inspect
import dm_ai_module

print('dm_ai_module file:', getattr(dm_ai_module, '__file__', None))
func = getattr(dm_ai_module.ActionGenerator, 'generate_legal_actions', None)
print('func object:', func)
try:
    src = inspect.getsource(func)
    print('--- source preview ---')
    for i, line in enumerate(src.splitlines()[:40]):
        print(f'{i+1:03}: {line}')
except Exception as e:
    print('could not get source:', e)
