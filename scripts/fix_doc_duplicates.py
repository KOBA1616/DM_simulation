#!/usr/bin/env python3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPLACEMENTS = [
    ("Specs", "Specs"),
    ("docs/Specs", "docs/Specs"),
    ("archive/docs/archive/docs", "archive/docs"),
    ("archive", "archive"),
    ("guides", "guides"),
    ("spell", "spell"),
    ("docs/guides", "docs/guides"),
    ("docs/spell", "docs/spell"),
]

def main():
    p = subprocess.run(["git", "ls-files", "-m"], cwd=ROOT, capture_output=True, text=True)
    files = [l.strip() for l in p.stdout.splitlines() if l.strip()]
    changed = []
    for f in files:
        fp = ROOT / f
        if not fp.exists():
            continue
        try:
            txt = fp.read_text(encoding='utf-8')
        except Exception:
            continue
        new = txt
        for a, b in REPLACEMENTS:
            new = new.replace(a, b)
        if new != txt:
            fp.write_text(new, encoding='utf-8')
            changed.append(f)
    print("FIXED:\n" + "\n".join(changed))

if __name__ == '__main__':
    main()
