#!/usr/bin/env python3
import sys, json, argparse

"""
Compare seq_dump events (seq_head) between two H2H JSON-line logs.
Usage: python tools/compare_seq_dump.py native.log pyfb.log
Writes diffs to stdout; returns exit code 0 if identical.
"""

def extract_seq_heads(path):
    out = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'H2H_JSON:' not in line:
                continue
            try:
                j = line.split('H2H_JSON:',1)[1].strip()
                ev = json.loads(j)
            except Exception:
                continue
            if ev.get('event') == 'seq_dump':
                out.append((ev.get('index'), ev.get('seq_head')))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_a')
    p.add_argument('log_b')
    args = p.parse_args()
    a = dict(extract_seq_heads(args.log_a))
    b = dict(extract_seq_heads(args.log_b))
    keys = sorted(set(a.keys()) | set(b.keys()))
    mismatches = []
    for k in keys:
        sa = a.get(k)
        sb = b.get(k)
        if sa != sb:
            mismatches.append((k, sa, sb))
    if not mismatches:
        print("OK: all seq_dump seq_head match")
        return 0
    else:
        print(f"Mismatches found: {len(mismatches)}")
        for k,sa,sb in mismatches:
            print(f"--- index={k} ---")
            print(f"A: {sa}")
            print(f"B: {sb}\n")
        return 2

if __name__ == '__main__':
    sys.exit(main())
