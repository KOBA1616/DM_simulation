#!/usr/bin/env python3
"""Check ONNX Runtime runtime vs build information to help diagnose API/ABI mismatches.

Outputs a short report to `reports/ort_runtime_check.txt` and prints a summary to stdout.
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'ort_runtime_check.txt'

def find_cmake_cache():
    candidates = [ROOT / 'build' / 'CMakeCache.txt', ROOT / 'build-ninja' / 'CMakeCache.txt', ROOT / 'CMakeCache.txt']
    for p in candidates:
        if p.exists():
            return p
    return None

def parse_cmake_for_ort(cache_path: Path):
    info = {}
    try:
        for ln in cache_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            if 'ONNXRUNTIME' in ln.upper() or 'ONNX' in ln.upper():
                # crude capture
                parts = ln.split('=')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = parts[-1].strip()
                    info[key] = val
    except Exception:
        pass
    return info

def inspect_python_onnxruntime():
    out = {}
    try:
        import importlib
        mod = importlib.import_module('onnxruntime')
        out['onnxruntime.__version__'] = getattr(mod, '__version__', None)
        pkg_path = Path(mod.__file__).parent
        out['onnxruntime.path'] = str(pkg_path)
        files = [p.name for p in pkg_path.iterdir() if p.is_file()]
        out['onnxruntime.files'] = files
    except Exception as e:
        out['error'] = str(e)
    return out

def locate_onnxruntime_dlls(root: Path):
    dlls = []
    for p in root.rglob('onnxruntime*.dll'):
        dlls.append(str(p))
    return dlls

def main():
    report = {}
    cache = find_cmake_cache()
    report['cmake_cache'] = str(cache) if cache is not None else None
    if cache:
        report['cmake_info'] = parse_cmake_for_ort(cache)

    py_info = inspect_python_onnxruntime()
    report['python_onnxruntime'] = py_info

    # search for onnxruntime dlls in common build locations
    dll_candidates = locate_onnxruntime_dlls(ROOT)
    report['found_onnxruntime_dlls'] = dll_candidates

    # write report
    try:
        OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        print('Failed to write report', OUT_PATH)

    # Summary
    print('ORT runtime check written to', OUT_PATH)
    if py_info.get('error'):
        print('onnxruntime import error:', py_info.get('error'))
    else:
        print('onnxruntime package version:', py_info.get('onnxruntime.__version__'))
        print('onnxruntime package path:', py_info.get('onnxruntime.path'))
    if report['cmake_cache']:
        print('CMakeCache found at', report['cmake_cache'])
    if dll_candidates:
        print('Found onnxruntime DLLs (sample):')
        for d in dll_candidates[:5]:
            print('  -', d)

if __name__ == '__main__':
    main()
