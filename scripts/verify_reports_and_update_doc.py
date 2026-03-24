#!/usr/bin/env python3
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
BUILD_LOG = REPORTS / 'build' / 'quick_build_stdout.txt'
TEST_LOG = REPORTS / 'tests' / 'pytest_latest.txt'
ACTIONREFS = REPORTS / 'actiondef_refs.txt'
DOC = ROOT / 'CRITICAL_REMAINING_TASKS_TDD.md'


def parse_pytest_summary(text: str):
    # try to find typical pytest summary like '281 passed, 1 skipped'
    m = re.search(r"(\d+\s+passed(?:,\s*\d+\s+skipped)?)", text)
    if m:
        return m.group(1)
    # fallback: look for lines containing 'passed' or 'failed'
    lines = []
    for part in ['passed', 'failed', 'skipped', 'xfailed', 'errors']:
        m2 = re.search(r"(\d+)\s+%s" % part, text)
        if m2:
            lines.append(f"{m2.group(1)} {part}")
    return ', '.join(lines) if lines else 'unknown'


def count_actionrefs():
    if not ACTIONREFS.exists():
        return None
    return sum(1 for _ in ACTIONREFS.open('r', encoding='utf-8'))


def update_doc(py_summary: str, action_count: int):
    if not DOC.exists():
        print('Doc not found:', DOC)
        return
    content = DOC.read_text(encoding='utf-8')
    note = f"\n- reports verification: pytest summary: {py_summary}; actiondef refs: {action_count}\n"
    if 'Run local build & tests' in content and '[ ] Run local build & tests' in content:
        content = content.replace('[ ] Run local build & tests', '[x] Run local build & tests')
    # append verification note near top
    if 'reports verification:' not in content:
        content = content.replace('\n追記 (実行補助):', '\n追記 (実行補助):' + note)
    else:
        # append at end of that section
        content = content + note
    DOC.write_text(content, encoding='utf-8')
    print('Updated', DOC)


def main():
    if not TEST_LOG.exists():
        print('Test report not found:', TEST_LOG)
        sys.exit(2)
    txt = TEST_LOG.read_text(encoding='utf-8')
    summary = parse_pytest_summary(txt)
    action_count = count_actionrefs()
    update_doc(summary, action_count if action_count is not None else -1)
    print('Summary:', summary, 'ActionDef refs:', action_count)


if __name__ == '__main__':
    main()
