#!/usr/bin/env python3
"""
Align installed `onnxruntime` package to the version detected in CMakeCache/build files.

Behavior:
 - Scans CMakeCache.txt and build logs for strings that look like an ONNX Runtime version.
 - If a version is found, prints the `pip install onnxruntime==<version>` command to run.
 - With `--apply` will invoke pip (in the current Python) to install that version.

Usage:
  python scripts/align_onnxruntime.py       # dry-run, reports detected version
  python scripts/align_onnxruntime.py --apply  # attempt to pip install matched version

This script is conservative: it will not uninstall other packages and will ask for confirmation
before running `pip install` when `--apply` is used.
"""
from pathlib import Path
import re
import sys
import subprocess
import argparse


def detect_version_from_file(p: Path):
    try:
        txt = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None
    # Common patterns: 1.20.1 or v1.20.1 or 1.20
    m = re.search(r'onnxruntime[\-_ ]?[:= ]?v?(\d+\.\d+\.\d+)', txt, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r'ONNXRUNTIME[-_ ]?VERSION[:= ]?v?(\d+\.\d+\.\d+)', txt, re.IGNORECASE)
    if m2:
        return m2.group(1)
    # fallback: any 1.x.y appearing near ONNX/ORT mention
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        if 'ONNX' in line.upper() or 'ORT' in line.upper() or 'ONNXRUNTIME' in line.upper():
            m3 = re.search(r'v?(\d+\.\d+\.\d+)', line)
            if m3:
                return m3.group(1)
    return None


def find_version():
    root = Path('.').absolute()
    candidates = [root / 'CMakeCache.txt', root / 'build' / 'CMakeCache.txt', root / 'build-ninja' / 'CMakeCache.txt', root / 'build-msvc' / 'CMakeCache.txt']
    for p in candidates:
        if p.exists():
            v = detect_version_from_file(p)
            if v:
                return v, str(p)

    # Search common build files
    for d in ['build', 'build-ninja', 'build-msvc']:
        dd = root / d
        if not dd.exists():
            continue
        for p in dd.rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.txt', '.log', '.cmake', '.cache']:
                v = detect_version_from_file(p)
                if v:
                    return v, str(p)

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    v, src = find_version()
    if not v:
        print('Could not detect ONNX Runtime version from CMake/build files.')
        print('Run scripts/check_ort_mismatch.py to gather diagnostics.')
        sys.exit(1)

    print(f'Detected ONNX Runtime version: {v} (from {src})')
    cmd = [sys.executable, '-m', 'pip', 'install', f'onnxruntime=={v}']
    print('Recommended command:')
    print(' '.join(cmd))

    if args.apply:
        resp = input('Proceed to pip install this version into the active Python? [y/N]: ').strip().lower()
        if resp != 'y':
            print('Aborted.')
            return
        print('Running pip install...')
        try:
            subprocess.check_call(cmd)
            print('Install completed. Re-run your tests.')
        except subprocess.CalledProcessError as e:
            print('pip install failed:', e)
            sys.exit(2)


if __name__ == '__main__':
    main()
