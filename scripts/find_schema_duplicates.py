"""Find duplicate or overlapping constant definitions between schema_config and consts.

Produces a Markdown report at `reports/duplicate_constants_report.md`.
"""
import inspect
import importlib
import json
from pathlib import Path
from typing import Any
import ast


def is_sequence(obj: Any) -> bool:
    return isinstance(obj, (list, tuple, set))


def load_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        print(f"Failed to import {name}: {e}")
        return None


def extract_constants(mod):
    consts = {}
    # If mod is a mapping namespace (from runpy), try dict-like access
    if isinstance(mod, dict):
        items = mod.items()
    else:
        items = [(k, getattr(mod, k)) for k in dir(mod) if not k.startswith('__')]

    for k, v in items:
        if k.startswith('__'):
            continue
        if is_sequence(v) or isinstance(v, (int, str, dict)):
            consts[k] = v
    return consts


def extract_constants_from_source(path: Path):
    """Statically parse top-level Assign nodes and extract literal lists/dicts."""
    results = {}
    try:
        src = path.read_text(encoding='utf-8')
    except Exception:
        return results
    try:
        tree = ast.parse(src)
    except Exception:
        return results

    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Only handle simple name targets
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                items = []
                for el in value.elts:
                    if isinstance(el, ast.Constant):
                        items.append(el.value)
                results[name] = items
            elif isinstance(value, ast.Dict):
                keys = []
                for k in value.keys:
                    if isinstance(k, ast.Constant):
                        keys.append(k.value)
                results[name] = {k: None for k in keys}
            elif isinstance(value, ast.Constant):
                results[name] = value.value
    return results


def summarize_overlap(a: dict, b: dict):
    overlaps = []
    for ka, va in a.items():
        for kb, vb in b.items():
            if is_sequence(va) and is_sequence(vb):
                set_a = set(va)
                set_b = set(vb)
                inter = set_a & set_b
                if inter:
                    overlaps.append((ka, kb, sorted(inter)))
            elif isinstance(va, dict) and isinstance(vb, dict):
                # compare keys
                inter = set(va.keys()) & set(vb.keys())
                if inter:
                    overlaps.append((ka, kb, sorted(inter)))
            elif va == vb:
                overlaps.append((ka, kb, [va]))
    return overlaps


def main():
    # Try importing via workspace package path; fall back to loading by file path
    try:
        schema_mod = load_module('dm_toolkit.gui.editor.schema_config')
        consts_mod = load_module('dm_toolkit.consts')
    except Exception:
        schema_mod = None
        consts_mod = None

    # If package import fails in test environment, load by executing file directly
    if not schema_mod:
        # Try static analysis fallback
        schema_path = Path('dm_toolkit/gui/editor/schema_config.py')
        if schema_path.exists():
            schema_consts = extract_constants_from_source(schema_path)
            class SimpleMod: pass
            sm = SimpleMod()
            for k, v in schema_consts.items():
                setattr(sm, k, v)
            schema_mod = sm
        else:
            schema_mod = None

    if not consts_mod:
        consts_path = Path('dm_toolkit/consts.py')
        if consts_path.exists():
            consts_consts = extract_constants_from_source(consts_path)
            class SimpleMod: pass
            cm = SimpleMod()
            for k, v in consts_consts.items():
                setattr(cm, k, v)
            consts_mod = cm
        else:
            consts_mod = None

    report_lines = []
    report_lines.append('# Duplicate/Overlapping Constants Report\n')

    if not schema_mod or not consts_mod:
        report_lines.append('Could not import modules; aborting.')
    else:
        a = extract_constants(schema_mod)
        b = extract_constants(consts_mod)

        report_lines.append('## Candidates in schema_config.py\n')
        for k in sorted(a.keys()):
            report_lines.append(f'- `{k}`: {type(a[k]).__name__}')

        report_lines.append('\n## Candidates in dm_toolkit/consts.py\n')
        for k in sorted(b.keys()):
            report_lines.append(f'- `{k}`: {type(b[k]).__name__}')

        report_lines.append('\n## Overlaps by value\n')
        overlaps = summarize_overlap(a, b)
        if overlaps:
            for ka, kb, items in overlaps:
                report_lines.append(f'- `{ka}` vs `{kb}`: {items}')
        else:
            report_lines.append('No overlapping sequence values detected.')

    out_dir = Path('reports')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'duplicate_constants_report.md'
    out_file.write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'Wrote report to {out_file}')


if __name__ == '__main__':
    main()
