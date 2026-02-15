import importlib, sys, os
print('PY:', sys.executable)
print('CWD:', os.getcwd())
print('FILES:', os.listdir('.'))
print('SYSPATH(before):', sys.path[:5])
# Ensure project root is on sys.path so top-level dm_ai_module.py is importable
proj_root = os.path.abspath(os.getcwd())
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
print('SYSPATH(after):', sys.path[:5])
try:
    m = importlib.import_module('dm_ai_module')
    print('MODULE_FILE:', getattr(m, '__file__', None))
    print('HAS_JsonLoader:', hasattr(m, 'JsonLoader'))
    print('SAMPLE_ATTRS:', sorted([x for x in dir(m) if not x.startswith('_')])[:80])
    # Instantiate GameInstance and inspect execute_action availability
    try:
        gi = m.GameInstance(0)
        print('GameInstance type:', type(gi))
        print('has execute_action on instance:', hasattr(gi, 'execute_action'))
        try:
            print('execute_action attr:', getattr(gi, 'execute_action'))
        except Exception:
            pass
        print('has resolve_action on instance:', hasattr(gi, 'resolve_action'))
        # Inspect class-level attribute presence
        try:
            print('has execute_action on class:', hasattr(m.GameInstance, 'execute_action'))
            try:
                print('class execute_action attr:', getattr(m.GameInstance, 'execute_action'))
            except Exception:
                pass
        except Exception:
            pass
    except Exception as e:
        print('GameInstance instantiation error:', e)
except Exception as e:
    print('IMPORT ERROR:', e)
    import traceback
    traceback.print_exc()
