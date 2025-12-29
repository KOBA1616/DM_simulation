#!/usr/bin/env python3
import os, re, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDE = {'.venv', 'venv', 'archive', 'build', 'build_debug', 'build_ci_test', '.mypy_cache', '.git', 'third_party', 'tmp_pr'}
EXTS = {'.py', '.json', '.md', '.yaml', '.yml', '.txt'}
pattern = re.compile(r"(?P<quote>['\"])actions(?P=quote)|\bactions\b")
report = []

for root, dirs, files in os.walk(ROOT):
    # filter dirs
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for fn in files:
        if pathlib.Path(fn).suffix.lower() not in EXTS:
            continue
        path = pathlib.Path(root) / fn
        try:
            txt = path.read_text(encoding='utf-8')
        except Exception:
            continue
        lines = txt.splitlines()
        hits = []
        for i, line in enumerate(lines, start=1):
            if pattern.search(line):
                # capture context
                start = max(0, i-3)
                end = min(len(lines), i+2)
                ctx = '\n'.join(f"{ln+1:5d}: {lines[ln]}" for ln in range(start, end))
                hits.append({'line': i, 'line_text': line.strip(), 'context': ctx})
        if hits:
            report.append({'path': str(path), 'count': len(hits), 'hits': hits})

report_path = ROOT / 'archive' / 'find_actions_report.txt'
report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open('w', encoding='utf-8') as f:
    f.write(f'Report generated for project root: {ROOT}\n')
    f.write(f'Files with matches: {len(report)}\n\n')
    for r in report:
        f.write(f"FILE: {r['path']} (matches: {r['count']})\n")
        for h in r['hits']:
            f.write(h['context'] + '\n')
        f.write('\n' + ('-'*60) + '\n\n')

# print concise summary
print('Report written to:', report_path)
print('Files matched:', len(report))
for r in report[:50]:
    print('-', r['path'], f"({r['count']})")

if len(report) > 50:
    print('... (truncated list)')

sys.exit(0)
