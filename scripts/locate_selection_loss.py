import re
from pathlib import Path

p = Path('logs/trace_combined.txt')
if not p.exists():
    print('trace_combined.txt not found')
    raise SystemExit(1)

text = p.read_text(encoding='utf-8', errors='ignore')
lines = text.splitlines()

def show_window(idx, w=40):
    start = max(0, idx-5)
    end = min(len(lines), idx+w)
    print('\n--- window around line', idx, '---')
    for j in range(start, end):
        print(f'{j:05d}: {lines[j]}')

hits = []
for i, line in enumerate(lines):
    if 'var_SELECTED_TARGETS' in line or 'var_SELECTED' in line:
        hits.append(i)

print(f'Found {len(hits)} lines mentioning var_SELECTED*')
for idx in hits:
    show_window(idx, 60)

# find DIAG_MOVE_MISS and RESOLVED_TARGETS
for i, line in enumerate(lines):
    if 'DIAG_MOVE_MISS' in line or 'RESOLVED_TARGETS' in line or 'RESOLVED_COUNT_AS_TARGETS' in line:
        show_window(i, 40)

print('\nDone')
