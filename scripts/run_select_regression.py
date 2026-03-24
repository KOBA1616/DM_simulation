#!/usr/bin/env python3
"""Run SELECT-related regression tests repeatedly and collect logs.

Usage: python scripts/run_select_regression.py -n 5 --log-dir=logs/select_regression
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SELECT_TESTS = [
    'tests/test_transition_reproducer.py',
    'tests/test_card1_hand_quality.py',
]


def run_once(iter_idx, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f'select_run_{iter_idx}.log'
    failed_test = None
    with out.open('w', encoding='utf-8') as f:
        f.write(f'Iteration {iter_idx}\n')
        for t in SELECT_TESTS:
            f.write('\n=== RUN TEST: ' + t + ' ===\n')
            proc = subprocess.run([sys.executable, '-m', 'pytest', t, '--maxfail=1', '-q'], capture_output=True, text=True)
            f.write(proc.stdout)
            f.write(proc.stderr)
            f.write(f'EXIT_CODE: {proc.returncode}\n')
            if proc.returncode != 0:
                f.write('--- TEST FAILED, STOPPING ITERATION ---\n')
                failed_test = t
                return False, out, failed_test
    return True, out, failed_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--iterations', type=int, default=5)
    p.add_argument('--log-dir', default='logs/select_regression')
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    summary = []
    for i in range(1, args.iterations + 1):
        ok, path, failed_test = run_once(i, log_dir)
        summary.append({
            'iteration': i,
            'ok': ok,
            'log_path': str(path),
            'failed_test': failed_test,
        })
        print(f'Iteration {i}: {"PASS" if ok else "FAIL"} -> {path}')
        if failed_test:
            print(f'  Failed test: {failed_test}')
        if not ok:
            break

    # 再発防止: 監視運用ではコンソール出力だけだと履歴比較が難しいため、
    # machine-readable な summary.json を常に出力して失敗手順の自動化に繋げる。
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
        extra = f" failed_test={row['failed_test']}" if row['failed_test'] else ''
        print(f"  Iter {row['iteration']}: {'PASS' if row['ok'] else 'FAIL'} ({row['log_path']}){extra}")
    print(f'Wrote summary: {summary_path}')

    if any(not row['ok'] for row in summary):
        sys.exit(1)


if __name__ == '__main__':
    main()
