#!/usr/bin/env python3
"""Run `scripts/run_native_onnx_loader.py` repeatedly and report the first failure.
Usage: python scripts/repeat_native_loader.py --count 100
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=100)
    args = p.parse_args()

    script = Path(__file__).resolve().parent / 'run_native_onnx_loader.py'
    if not script.exists():
        print('run_native_onnx_loader.py not found', file=sys.stderr)
        return 2

    for i in range(1, args.count + 1):
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        print(f'Iter {i}: EXIT {proc.returncode}')
        if proc.stdout:
            print('STDOUT:', proc.stdout.strip())
        if proc.stderr:
            print('STDERR:', proc.stderr.strip())
        if proc.returncode != 0:
            print('Failure detected on iteration', i)
            # Save outputs for inspection
            outdir = Path('logs') / 'native_repeat'
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f'iter_{i}_stdout.txt').write_text(proc.stdout or '')
            (outdir / f'iter_{i}_stderr.txt').write_text(proc.stderr or '')
            (outdir / f'iter_{i}_exit.txt').write_text(str(proc.returncode))
            return 1

    print('All iterations completed without failure')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
