#!/usr/bin/env python3
"""List all references to 'ActionDef' / 'action_def' in the repository.

Writes a report to `reports/actiondef_refs.txt` and prints a short summary.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'actiondef_refs.txt'

PATTERNS = [re.compile(r"\bActionDef\b"), re.compile(r"action_def")]

# Only scan these text-based extensions to avoid binary/large files
ALLOWED_EXTS = {'.py', '.cpp', '.hpp', '.c', '.h', '.md', '.txt', '.json', '.yaml', '.yml', '.ini', '.toml'}

IGNORES = {'.git', 'build', 'build-msvc', 'build-ninja', 'bin', 'archive', '.venv', 'venv', '__pycache__'}

def should_ignore(path: Path) -> bool:
    for p in path.parts:
        if p in IGNORES:
            return True
    return False

def scan() -> dict:
    results = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune ignored directories early to avoid walking large trees
        dirnames[:] = [d for d in dirnames if d not in IGNORES and not d.startswith('.')]
        pdir = Path(dirpath)
        if should_ignore(pdir):
            continue
        for fname in filenames:
            # skip binary files by extension
            if fname.endswith(('.pyc', '.pyd', '.so', '.dll', '.exe')):
                continue
            if Path(fname).suffix.lower() not in ALLOWED_EXTS:
                continue
            fpath = pdir / fname
            try:
                # Skip very large files to avoid hangs
                try:
                    if fpath.stat().st_size > 1_000_000:
                        continue
                except Exception:
                    # if stat fails, skip
                    continue
                try:
                    text = fpath.read_text(encoding='utf-8', errors='ignore')
                except BaseException:
                    # catch KeyboardInterrupt and other BaseExceptions to avoid aborting
                    continue
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                for pat in PATTERNS:
                    if pat.search(line):
                        results.setdefault(str(fpath.relative_to(ROOT)), []).append((i, line.strip()))
                        break
    return results

def main():
    res = scan()
    with OUT_PATH.open('w', encoding='utf-8') as f:
        if not res:
            f.write('NO_MATCHES\n')
            print('No ActionDef references found.')
            return
        for fp, hits in sorted(res.items()):
            f.write(f'File: {fp}\n')
            for ln, txt in hits:
                f.write(f'  {ln}: {txt}\n')
            f.write('\n')
    print(f'Found {sum(len(v) for v in res.values())} matches in {len(res)} files. Report: {OUT_PATH}')

if __name__ == '__main__':
    main()
