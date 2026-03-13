#!/usr/bin/env python3
"""Audit remaining raw `.connect(` usages under editor forms and produce a priority report.

Usage:
  python scripts/audit_connects_remaining.py [root_dir] [--out report.json]

Default root_dir: dm_toolkit/gui/editor/forms
"""
import os
import re
import json
import argparse


def find_connects(root):
    pattern = re.compile(r"\.connect\s*\(")
    files_counts = {}
    for dirpath, dirs, files in os.walk(root):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    text = fh.read()
            except Exception:
                continue
            cnt = len(pattern.findall(text))
            if cnt:
                files_counts[path.replace('\\\\','/')]=cnt
    return files_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='dm_toolkit/gui/editor/forms')
    parser.add_argument('--out', default='scripts/connect_audit_report.json')
    args = parser.parse_args()

    files_counts = find_connects(args.root)
    sorted_items = sorted(files_counts.items(), key=lambda x: x[1], reverse=True)
    total = sum(files_counts.values())

    print(f'Total raw .connect occurrences: {total}')
    for path, cnt in sorted_items:
        print(f'{cnt:4d} {path}')

    report = {'total': total, 'files': [{'path': p, 'count': c} for p, c in sorted_items]}
    try:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f'Report written to {args.out}')
    except Exception as e:
        print(f'Failed to write report: {e}')


if __name__ == '__main__':
    main()
