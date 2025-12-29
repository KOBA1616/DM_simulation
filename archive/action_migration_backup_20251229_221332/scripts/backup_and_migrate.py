#!/usr/bin/env python3
import os, sys, shutil, json, subprocess, datetime
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = os.path.join(ROOT, 'archive', f'action_migration_backup_{now}')
os.makedirs(backup_dir, exist_ok=True)
print('Backup dir:', backup_dir)
matched = []
for root, dirs, files in os.walk(ROOT):
    # skip archive folder itself
    if os.path.commonpath([root, backup_dir]) == backup_dir:
        continue
    for fn in files:
        if not fn.lower().endswith(('.json', '.py', '.md')):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                s = f.read()
        except Exception:
            continue
        if '"actions"' in s or "'actions'" in s:
            matched.append(path)
            dest = os.path.join(backup_dir, os.path.relpath(path, ROOT))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(path, dest)

print(f'Found {len(matched)} files containing "actions". Backed up copies created.')
for p in matched:
    print('-', p)

# Run migration script over repository root to convert actions->commands in JSON files
migrate_script = os.path.join(ROOT, 'scripts', 'migrate_actions_to_commands.py')
if not os.path.exists(migrate_script):
    print('Migration script not found:', migrate_script)
    sys.exit(2)
print('\nRunning migration script over repository root...')
proc = subprocess.run([sys.executable, migrate_script, ROOT], cwd=ROOT)
print('Migration exit code:', proc.returncode)

# Report remaining files still containing 'actions' after migration
remaining = []
for root, dirs, files in os.walk(ROOT):
    for fn in files:
        if not fn.lower().endswith(('.json', '.py', '.md')):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                s = f.read()
        except Exception:
            continue
        if '"actions"' in s or "'actions'" in s:
            remaining.append(path)

print(f'\nRemaining files containing "actions": {len(remaining)}')
for p in remaining:
    print('-', p)

# Exit code reflects if any remaining
sys.exit(0 if len(remaining)==0 else 3)
