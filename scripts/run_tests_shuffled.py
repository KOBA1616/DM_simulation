#!/usr/bin/env python3
"""Run test files in random file-order for multiple iterations to detect order-dependency.

Usage: python scripts/run_tests_shuffled.py -n 10
"""
import argparse
import glob
import json
import random
import subprocess
import sys
from pathlib import Path


def find_test_files():
    # collect top-level test files under tests/
    files = sorted([p for p in glob.glob('tests/**/*.py', recursive=True) if Path(p).name.startswith('test_')])
    return files


def run_iteration(files, iter_idx, log_dir):
    random.shuffle(files)
    log_path = log_dir / f"test_shuffled_run{iter_idx}.log"
    any_fail = False
    failed_file = None
    executed_files = []
    with log_path.open('w', encoding='utf-8') as f:
        f.write(f"Iteration {iter_idx}\n")
        f.write("Order:\n")
        for fi in files:
            f.write(fi + "\n")
        f.write('\n')
        for fi in files:
            f.write(f"=== RUN FILE: {fi} ===\n")
            executed_files.append(fi)
            proc = subprocess.run([sys.executable, '-m', 'pytest', fi, '--maxfail=1', '-q'], capture_output=True, text=True)
            f.write(proc.stdout)
            f.write(proc.stderr)
            f.write(f"EXIT_CODE: {proc.returncode}\n\n")
            if proc.returncode != 0:
                any_fail = True
                failed_file = fi
                f.write(f"--- STOPPING ITERATION {iter_idx} DUE TO FAILURE ---\n")
                break
    return any_fail, log_path, failed_file, files, executed_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--iterations', type=int, default=10)
    p.add_argument('--log-dir', default='.', help='Directory to write iteration logs')
    args = p.parse_args()

    files = find_test_files()
    if not files:
        print('No test files found under tests/ matching test_*.py', file=sys.stderr)
        sys.exit(2)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for i in range(1, args.iterations + 1):
        fail, log_path, failed_file, run_order, executed_files = run_iteration(files[:], i, log_dir)
        summary.append({
            'iteration': i,
            'ok': (not fail),
            'log_path': str(log_path),
            'failed_file': failed_file,
            'run_order': run_order,
            'executed_files': executed_files,
        })
        print(f"Iteration {i}: {'PASS' if not fail else 'FAIL'} -> {log_path}")
        if failed_file:
            print(f"  Failed file: {failed_file}")
        if fail:
            print('Failure detected — stopping further iterations.')
            break

    # 再発防止: ランダム順序試験の結果はコンソール出力だけだと比較が難しいため、
    # 反復結果・失敗ファイル・実行順を summary.json として保存する。
    summary_path = log_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'iterations_requested': args.iterations,
                'iterations_executed': len(summary),
                'results': summary,
            },
            ensure_ascii=False,
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )

    print('\nSummary:')
    for row in summary:
        extra = f" failed_file={row['failed_file']}" if row['failed_file'] else ''
        print(f"  Iter {row['iteration']}: {'PASS' if row['ok'] else 'FAIL'} ({row['log_path']}){extra}")
    print(f"Wrote summary: {summary_path}")

    # exit nonzero if any iteration failed
    if any(not row['ok'] for row in summary):
        sys.exit(1)


if __name__ == '__main__':
    main()
