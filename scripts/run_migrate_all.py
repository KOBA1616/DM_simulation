"""Run migrate_cost_reduction_ids.py on all JSON files under data/ and report results.

Usage: python scripts/run_migrate_all.py
"""
import glob
import subprocess
import sys
from pathlib import Path

root = Path('data')
if not root.exists():
    print('data/ directory not found; aborting')
    sys.exit(2)

files = sorted(glob.glob('data/**/*.json', recursive=True))
if not files:
    print('No JSON files found under data/')
    sys.exit(0)

success = []
failed = []
for p in files:
    try:
        print('Processing', p)
        subprocess.check_call([sys.executable, 'scripts/migrate_cost_reduction_ids.py', p])
        success.append(p)
    except subprocess.CalledProcessError:
        print('FAILED', p)
        failed.append(p)

print('\nSummary:')
print('  processed:', len(files))
print('  success :', len(success))
print('  failed  :', len(failed))
if failed:
    print('Failed files:')
    for f in failed:
        print('  ', f)
sys.exit(0)
