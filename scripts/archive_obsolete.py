import os
import shutil
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARCHIVE = os.path.join(ROOT, 'archive', 'cleanup_2026-02-09')
os.makedirs(ARCHIVE, exist_ok=True)

patterns = [
    'tmpclaude-*',
    'tmp_selfplay_*.log*',
    'tmp_selfplay_long.log*',
    'gui_*.log',
    'build_quick.log',
    'debug_*.py',
    'find_attack_phase.py',
    'verify_fixes.py',
    'test_*.py',
    'test_*.ps1',
]

moved = []
for p in patterns:
    matches = glob.glob(os.path.join(ROOT, p))
    for src in matches:
        try:
            dest = os.path.join(ARCHIVE, os.path.basename(src))
            shutil.move(src, dest)
            moved.append(dest)
        except Exception:
            pass

print('Archived:', len(moved), 'items')
for m in moved:
    print(' -', m)
