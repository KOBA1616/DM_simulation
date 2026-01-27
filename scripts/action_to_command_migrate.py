#!/usr/bin/env python3
"""
Conservative migration helper (Action -> Command).
- By default performs a dry-run and reports candidate files/occurrences.
- Replaces only `generate_legal_actions(` -> `generate_legal_commands(` when `--apply` is passed.
- If `--backup` is passed with `--apply`, writes a `.bak` copy before modifying.

Use carefully and review diffs before applying.
"""
import argparse
import re
from pathlib import Path

DEF_PAT = re.compile(r"\bgenerate_legal_actions\b")

def scan(root: Path):
    files = []
    for p in root.rglob('*.py'):
        # skip virtualenvs, build dirs, tests generated artifacts
        if any(part in ('venv', '.venv', 'build', 'dist', 'archive', 'bin') for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding='utf-8')
        except Exception:
            continue
        if DEF_PAT.search(txt):
            files.append(p)
    return files


def replace_in_file(path: Path, apply: bool, backup: bool):
    txt = path.read_text(encoding='utf-8')
    new = txt.replace('generate_legal_actions(', 'generate_legal_commands(')
    if new == txt:
        return False, []
    # collect context lines
    contexts = []
    for i, line in enumerate(txt.splitlines(), start=1):
        if 'generate_legal_actions' in line:
            contexts.append((i, line.strip()))
    if apply:
        if backup:
            path.with_suffix(path.suffix + '.bak').write_text(txt, encoding='utf-8')
        path.write_text(new, encoding='utf-8')
    return True, contexts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='Repository root')
    p.add_argument('--dry-run', action='store_true', default=False)
    p.add_argument('--apply', action='store_true', default=False)
    p.add_argument('--backup', action='store_true', default=False)
    args = p.parse_args()

    root = Path(args.root).resolve()
    files = scan(root)
    print(f'Found {len(files)} Python files containing `generate_legal_actions`')
    total = 0
    for f in files:
        changed, contexts = replace_in_file(f, apply=(args.apply and not args.dry_run), backup=args.backup)
        if changed:
            total += 1
            print(f"-- {f} -> will{' ' if args.dry_run else ' '}be modified")
            for ln, snippet in contexts:
                print(f"   {f}:{ln}: {snippet}")
    print(f'Files that would be/are modified: {total}')
    if args.apply and not args.dry_run:
        print('\nApply complete.')
    else:
        print('\nDry-run complete. Run with `--apply --backup` to apply changes (review .bak files).')

if __name__ == '__main__':
    main()
