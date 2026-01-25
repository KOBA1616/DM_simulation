"""
Scan repository for symbols referenced from `dm_ai_module` usage and report which ones are missing
from the actual imported `dm_ai_module` module. Outputs a short report to stdout and
exits with code 0.

Run:
    python scripts/check_native_symbols.py

"""
import ast
import os
import re
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def scan_files():
    names = set()
    pattern_attr = re.compile(r"dm_ai_module\.(\w+)")
    for p in ROOT.rglob('*.py'):
        # skip venv and hidden dirs
        if any(part.startswith('.') for part in p.parts):
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        # from dm_ai_module import X, Y
        for m in re.finditer(r"from\s+dm_ai_module\s+import\s+([\w\s,]+)", text):
            cols = m.group(1)
            for n in [c.strip() for c in cols.split(',') if c.strip()]:
                names.add(n)
        # attribute style dm_ai_module.X
        for m in pattern_attr.finditer(text):
            names.add(m.group(1))
    return sorted(names)


def check_module(names):
    missing = []
    present = []
    try:
        m = importlib.import_module('dm_ai_module')
    except Exception as e:
        print('ERROR: could not import dm_ai_module:', e)
        return None, None
    for n in names:
        if hasattr(m, n):
            present.append(n)
        else:
            missing.append(n)
    return present, missing


def main():
    names = scan_files()
    print(f"Scanned {len(names)} symbol candidates referencing dm_ai_module")
    present, missing = check_module(names)
    if present is None:
        return 2
    print('\nPresent symbols (sample up to 40):')
    print(', '.join(present[:40]))
    print('\nMissing symbols (sample up to 100):')
    print(', '.join(missing[:200]) or '<none>')
    # write report
    out = ROOT / 'docs' / 'missing_native_symbols.md'
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write('# Missing native symbols report\n\n')
            f.write('Detected references to `dm_ai_module` symbols and whether they are present.\n\n')
            f.write('## Present\n\n')
            for n in present:
                f.write(f'- {n}\n')
            f.write('\n## Missing\n\n')
            for n in missing:
                f.write(f'- {n}\n')
        print(f'Wrote report to {out}')
    except Exception as e:
        print('Failed to write report:', e)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
