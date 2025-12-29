#!/usr/bin/env python3
import os, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {'archive', '.venv', 'venv', 'build', 'build_debug', 'build_ci_test', '.mypy_cache', 'tmp_pr', 'third_party'}
modified_files = []

def should_skip(path: pathlib.Path):
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False

def strip_actions(obj):
    changed = False
    if isinstance(obj, dict):
        if 'actions' in obj:
            del obj['actions']
            changed = True
        for k, v in list(obj.items()):
            c = strip_actions(v)
            if c:
                changed = True
    elif isinstance(obj, list):
        for it in obj:
            c = strip_actions(it)
            if c:
                changed = True
    return changed

for root, dirs, files in os.walk(ROOT):
    # modify dirs in-place to skip excluded
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for fn in files:
        if not fn.lower().endswith('.json'):
            continue
        path = pathlib.Path(root) / fn
        if should_skip(path):
            continue
        try:
            text = path.read_text(encoding='utf-8')
            data = json.loads(text)
        except Exception:
            continue
        changed = strip_actions(data)
        if changed:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            modified_files.append(str(path))

print('Modified JSON files count:', len(modified_files))
for p in modified_files:
    print('-', p)
