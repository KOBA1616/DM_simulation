#!/usr/bin/env python3
"""Scan Python modules for ALL_CAPS list definitions and compare across files."""
import ast
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
files_to_scan = [
    ROOT / 'dm_toolkit' / 'consts.py',
    ROOT / 'dm_toolkit' / 'gui' / 'editor' / 'schema_config.py'
]

def extract_lists(path):
    res = {}
    try:
        src = path.read_text(encoding='utf-8')
    except Exception as e:
        return res
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    # Try to extract list/tuple literal constants
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        items = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                items.append(elt.value)
                        res[t.id] = {'kind': 'list', 'items': items}
                    else:
                        # Fallback: record that the name exists but value not a literal
                        seg = ast.get_source_segment(src, node.value)
                        res[t.id] = {'kind': 'expr', 'expr': seg}
    return res

all_defs = {}
for p in files_to_scan:
    all_defs[str(p.relative_to(ROOT))] = extract_lists(p)

# Build reverse index
value_index = defaultdict(list)
for fname, defs in all_defs.items():
    for name, meta in defs.items():
        if meta.get('kind') == 'list':
            for it in meta.get('items', []):
                value_index[(name, it)].append(fname)

# Report
print('Scanned files:')
for f in all_defs:
    print(' -', f)
print('\nTop-level ALL_CAPS lists found:')
for fname, defs in all_defs.items():
    print(f'\nFile: {fname}')
    for name, meta in defs.items():
        if meta.get('kind') == 'list':
            print(f'  {name}: {len(meta.get("items", []))} items')
        else:
            print(f'  {name}: expression alias -> {meta.get("expr")!r}')

# Detect lists with same name in multiple files
print('\nLists with same name in multiple files:')
name_locations = defaultdict(list)
for fname, defs in all_defs.items():
    for name in defs:
        name_locations[name].append(fname)
for name, locs in name_locations.items():
    if len(locs) > 1:
        print(f'  {name}: found in {locs}')

# Detect overlapping values across different list names/files
print('\nValues appearing in multiple lists:')
for (listname, value), files in sorted(value_index.items()):
    if len(set(files)) > 1:
        print(f'  Value {value!r} in list {listname} appears in files {sorted(set(files))}')

# Summary
print('\nSummary:')
for name, locs in name_locations.items():
    if len(locs) > 1:
        print(f' - Duplicate list name: {name} -> {locs}')

# Exit

