"""Gather diagnostics for ONNX Runtime mismatch between build and runtime.

Outputs to stdout and to `reports/ort_diagnostic.txt`.
"""
import sys
import platform
import os
from pathlib import Path
out = []
def w(s):
    out.append(str(s))
    print(s)

w('Python: ' + sys.version.replace('\n',' '))
w('Executable: ' + sys.executable)
w('Platform: ' + platform.platform())
w('Arch: ' + str(platform.architecture()))

# venv packages
try:
    import importlib
    pkg = importlib.import_module('onnxruntime')
    w('onnxruntime.__version__: ' + getattr(pkg, '__version__', 'unknown'))
except Exception as e:
    w('onnxruntime import failed: ' + repr(e))

# dm_ai_module info
try:
    import importlib, importlib.util
    sys.path.insert(0, str(Path('.').absolute()))
    dm = importlib.import_module('dm_ai_module')
    w('dm_ai_module loaded: ' + getattr(dm, '__file__', 'unknown'))
    # list presence of native loader
    w('has native_load_onnx: ' + str(hasattr(dm, 'native_load_onnx')))
except Exception as e:
    w('dm_ai_module import failed: ' + repr(e))

# Inspect CMakeCache.txt and build files for ONNX/ORT entries
root = Path('.').absolute()
candidates = [root / 'CMakeCache.txt', root / 'build' / 'CMakeCache.txt', root / 'build-ninja' / 'CMakeCache.txt', root / 'build-msvc' / 'CMakeCache.txt']
found = False
for p in candidates:
    if p.exists():
        found = True
        w('\n== Found CMakeCache: ' + str(p))
        try:
            data = p.read_text(encoding='utf-8', errors='ignore')
            for line in data.splitlines():
                if 'ONNX' in line.upper() or 'ORT' in line.upper() or 'ONNXRUNTIME' in line.upper():
                    w(line)
        except Exception as e:
            w('Failed to read ' + str(p) + ': ' + repr(e))
if not found:
    w('\nNo CMakeCache.txt found in expected locations.')

# Search build logs for ONNX/ORT mentions
w('\n== Searching build directories for ORT/ONNX references ==')
for d in ['build', 'build-ninja', 'build-msvc', 'bin']:
    dd = root / d
    if dd.exists():
        for p in dd.rglob('*'):
            try:
                if p.is_file() and p.suffix.lower() in ['.txt','.log','.cmake','.cache']:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                    if 'ONNX' in txt.upper() or 'ORT' in txt.upper() or 'ONNXRUNTIME' in txt.upper():
                        w(f'-- {p}:')
                        # print a few matching lines
                        for i,line in enumerate(txt.splitlines()):
                            if 'ONNX' in line.upper() or 'ORT' in line.upper() or 'ONNXRUNTIME' in line.upper():
                                w('   ' + line.strip())
                                if i>50:
                                    break
            except Exception:
                continue

# Save report
reports = root / 'reports'
reports.mkdir(exist_ok=True)
report_file = reports / 'ort_diagnostic.txt'
report_file.write_text('\n'.join(out), encoding='utf-8')
print('\nWrote report to', report_file)
