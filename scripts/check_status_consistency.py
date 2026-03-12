"""Check and optionally fix `status.md` test count to match `reports/tests/pytest_latest.txt`.

Usage:
  python scripts/check_status_consistency.py        # report status, exit nonzero if mismatch
  python scripts/check_status_consistency.py --fix  # update status.md when mismatch

Behavior:
- Parses `reports/tests/pytest_latest.txt` for a collected/tests summary (e.g. "collected 269 items" or "Total tests: 269").
- Parses `status.md` for a line like "- 結果: 269 passed, 0 failed" and updates the number of passed tests.
- Updates the Updated date at the top of `status.md` when --fix is used.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / 'reports' / 'tests' / 'pytest_latest.txt'
STATUS = ROOT / 'status.md'

if not REPORT.exists():
    print(f'ERROR: {REPORT} not found', file=sys.stderr)
    sys.exit(2)
if not STATUS.exists():
    print(f'ERROR: {STATUS} not found', file=sys.stderr)
    sys.exit(2)

text = REPORT.read_text(encoding='utf-8')
# Try to find 'collected N items' or 'Total tests: N' or 'collected N'
m = re.search(r'collected\s+(\d+)\s+items', text)
if m is None:
    m = re.search(r'collected\s+(\d+)', text)
if m is None:
    m = re.search(r'Total tests:\s*(\d+)', text)

if m is None:
    print('Could not find collected test count in pytest_latest.txt', file=sys.stderr)
    sys.exit(3)

collected = int(m.group(1))

s_text = STATUS.read_text(encoding='utf-8')
# Find a line like '- 結果: 269 passed, 0 failed' (Japanese '結果') or English 'Result:'
res_re = re.compile(r'(-\s*結果:\s*)([0-9]+)\s*passed(,\s*([0-9]+)\s*failed)?', re.IGNORECASE)
res_m = res_re.search(s_text)

if res_m:
    passed = int(res_m.group(2))
else:
    # fallback: search for 'Result: 269 passed'
    res_re2 = re.compile(r'(Result:\s*)([0-9]+)\s*passed', re.IGNORECASE)
    res_m2 = res_re2.search(s_text)
    if res_m2:
        passed = int(res_m2.group(2))
    else:
        print('Could not find passed count in status.md', file=sys.stderr)
        sys.exit(4)

print(f'pytest report: collected={collected}, status.md: passed={passed}')

if collected == passed:
    print('OK: counts match')
    sys.exit(0)

print('MISMATCH: counts differ')
if '--fix' not in sys.argv:
    print('Run with --fix to update status.md', file=sys.stderr)
    sys.exit(1)

# Perform fix: replace passed count and update Updated date
new_s = s_text
if res_m:
    new_s = res_re.sub(lambda m: f"{m.group(1)}{collected} passed, 0 failed", new_s, count=1)
else:
    new_s = res_re2.sub(lambda m: f"{m.group(1)}{collected} passed", new_s, count=1)

# Update '更新日:' line if present, else insert after title
upd_re = re.compile(r'(更新日:\s*).*')
if upd_re.search(new_s):
    new_s = upd_re.sub(f"\1{datetime.utcnow().strftime('%Y-%m-%d')}", new_s, count=1)
else:
    # insert after first line
    lines = new_s.splitlines()
    if len(lines) > 0:
        lines.insert(1, f"更新日: {datetime.utcnow().strftime('%Y-%m-%d')}")
        new_s = '\n'.join(lines)

STATUS.write_text(new_s, encoding='utf-8')
print(f'Updated {STATUS} to show {collected} passed and refreshed update date')
sys.exit(0)
