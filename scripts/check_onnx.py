import importlib, sys
sys.path.insert(0, r'c:\Users\ichirou\DM_simulation')
mod = importlib.import_module('tests.test_onnxruntime_version_alignment')
print('EXPECTED:', getattr(mod, 'EXPECTED_ORT_VERSION', '<missing>'))
import onnxruntime
print('runtime:', onnxruntime.__version__)
print('file:', getattr(onnxruntime, '__file__', None))
print('sys.path:')
for p in sys.path[:10]:
    print(' ', p)
