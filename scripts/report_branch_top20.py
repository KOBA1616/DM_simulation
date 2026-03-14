#!/usr/bin/env python3
"""Compute top-20 files by if/elif branch count and print as markdown.
"""
import os
import re
from collections import Counter

ROOT = r"C:\Users\ichirou\DM_simulation"
pattern_if = re.compile(r"\bif\b")
pattern_elif = re.compile(r"\belif\b")

counts = Counter()
file_counts = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    if any(p in dirpath for p in (".venv", "build", "build-msvc", "build-ninja", ".git", "__pycache__")):
        continue
    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            continue
        text_clean = re.sub(r"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"", '', text)
        lines = []
        for line in text_clean.splitlines():
            stripped = line.split('#', 1)[0]
            lines.append(stripped)
        src = '\n'.join(lines)
        ifs = len(pattern_if.findall(src))
        elifs = len(pattern_elif.findall(src))
        total = ifs + elifs
        if total > 0:
            rel = os.path.relpath(fp, ROOT).replace('\\', '/')
            file_counts[rel] = total
            counts['if'] += ifs
            counts['elif'] += elifs

total_files = len(file_counts)
total_branches = counts['if'] + counts['elif']

print(f"Branch baseline scan in: {ROOT}")
print(f"Files with branches: {total_files}")
print(f"Total if: {counts['if']}, elif: {counts['elif']}, total branches: {total_branches}")
print('\nTop 20 files by branch count:')
for fn, c in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {c:4d}  {fn}")

print('\nMarkdown table for plan insertion:')
print('| Rank | Branches | File |')
print('|---:|---:|---|')
for i, (fn, c) in enumerate(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:20], start=1):
    print(f'| {i} | {c} | {fn} |')

# Also write markdown table to reports/branch_top20.md for plan insertion
out_dir = os.path.join(ROOT, 'reports')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'branch_top20.md')
with open(out_path, 'w', encoding='utf-8') as of:
    of.write('# Branch Top-20 (generated)\n')
    of.write(f'Branch baseline scan in: {ROOT}\n\n')
    of.write(f'Files with branches: {total_files}\n')
    of.write(f'Total if: {counts["if"]}, elif: {counts["elif"]}, total branches: {total_branches}\n\n')
    of.write('| Rank | Branches | File |\n')
    of.write('|---:|---:|---|\n')
    for i, (fn, c) in enumerate(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:20], start=1):
        of.write(f'| {i} | {c} | {fn} |\n')
print(f"Wrote report to: {out_path}")
