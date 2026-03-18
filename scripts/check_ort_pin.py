#!/usr/bin/env python3
"""Check that installed onnxruntime matches the pinned version in requirements.txt

Exits with code 0 when matching, 1 when mismatch, 2 when onnxruntime not installed,
and 3 when requirements.txt does not pin onnxruntime.
"""
import re
import sys
from pathlib import Path


def read_pinned_version(req_path: Path):
    text = req_path.read_text(encoding='utf-8')
    for line in text.splitlines():
        m = re.match(r'\s*onnxruntime\s*==\s*([0-9.]+)\s*$', line)
        if m:
            return m.group(1)
    return None


def main():
    req = Path('requirements.txt')
    if not req.exists():
        print('requirements.txt not found', file=sys.stderr)
        return 3

    pinned = read_pinned_version(req)
    if not pinned:
        print('onnxruntime not pinned in requirements.txt', file=sys.stderr)
        return 3

    try:
        import onnxruntime as ort
    except Exception:
        print('onnxruntime not installed in current environment', file=sys.stderr)
        return 2

    inst = getattr(ort, '__version__', None) or getattr(ort, 'version', None)
    print(f'Pinned: {pinned}  Installed: {inst}')
    if inst is None:
        print('Could not determine installed onnxruntime version', file=sys.stderr)
        return 2
    if inst.split('+')[0] != pinned:
        print('VERSION_MISMATCH: Installed onnxruntime does not match pinned version', file=sys.stderr)
        return 1
    print('OK: onnxruntime matches pinned version')
    return 0


if __name__ == '__main__':
    sys.exit(main())
