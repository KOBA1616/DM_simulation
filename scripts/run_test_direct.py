import importlib.util
import traceback
import pathlib

print('Running test_generate_play_candidates_present directly')
try:
    test_path = pathlib.Path(__file__).parent.parent / 'tests' / 'test_generate_legal_commands.py'
    spec = importlib.util.spec_from_file_location('test_generate_legal_commands', str(test_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        print('EXCEPTION during module import:')
        traceback.print_exc()
        raise
    # Call the test function directly
    mod.test_generate_play_candidates_present()
    print('TEST PASSED (no crash)')
except AssertionError as e:
    print('TEST FAILED (assertion)')
    print(e)
except Exception:
    print('EXCEPTION while running test:')
    traceback.print_exc()
    raise
