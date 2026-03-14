#!/usr/bin/env python3
"""Count 'if' and 'elif' occurrences in Python files to measure branching.
Outputs total counts and top files by branch count.
"""
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(__file__))
pattern_if = re.compile(r"\bif\b")
pattern_elif = re.compile(r"\belif\b")

counts = Counter()
file_counts = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    # Skip virtualenv and build folders
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
        # Remove triple-quoted strings (simple heuristic)
        text_clean = re.sub(r"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"", '', text)
        # Remove single-line comments
        lines = []
        for line in text_clean.splitlines():
            stripped = line.split('#', 1)[0]
            lines.append(stripped)
        src = '\n'.join(lines)
        ifs = len(pattern_if.findall(src))
        elifs = len(pattern_elif.findall(src))
        total = ifs + elifs
        if total > 0:
            file_counts[os.path.relpath(fp, ROOT)] = total
            counts['if'] += ifs
            counts['elif'] += elifs

total_files = len(file_counts)
total_branches = counts['if'] + counts['elif']
print(f"Branch baseline scan in: {ROOT}")
print(f"Files with branches: {total_files}")
print(f"Total if: {counts['if']}, elif: {counts['elif']}, total branches: {total_branches}")
print("Top files by branch count:")
for fn, c in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {c:4d}  {fn}")

# Exit code 0
