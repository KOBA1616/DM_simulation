#!/usr/bin/env python3
"""
Annotate Python files that reference `ActionDef` to mark them as legacy.

This script is conservative: by default it runs in dry-run mode and writes
`reports/actiondef_refs_fixed.txt` listing files that would be annotated.
Use `--apply` to actually write backups (`.py.bak`) and prepend an annotation
header.

Usage:
  python scripts/annotate_actiondef_refs.py        # dry-run, writes report
  python scripts/annotate_actiondef_refs.py --apply  # apply changes
"""
from pathlib import Path
import argparse
import os

SEARCH_DIRS = ["dm_toolkit", "python", "tests"]
IGNORED_DIRS = {".venv", "build", "build-msvc", "build-ninja", "bin", "archive", "reports"}
HEADER = "# NOTE: This file references legacy 'ActionDef'. Prefer 'CommandDef' where possible.\n"
TRAILER = "\n# LEGACY_ACTIONDEF_REFERENCE: This file references 'ActionDef' (legacy). Consider migrating to 'CommandDef'.\n"


def annotate(root: Path, apply_changes: bool) -> list:
    modified = []
    for d in SEARCH_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if any(part in IGNORED_DIRS for part in p.parts):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if "ActionDef" in text:
                # avoid modifying a file if it already contains our marker
                rel = str(p.relative_to(root))
                if TRAILER.strip() in text or HEADER in text:
                    # already annotated
                    continue
                modified.append(rel)
                if apply_changes:
                    bak = p.with_suffix(p.suffix + ".bak")
                    bak.write_text(text, encoding="utf-8")
                    # append a non-intrusive trailer comment instead of prepending header
                    new_text = text + TRAILER
                    p.write_text(new_text, encoding="utf-8")
    return modified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (create .bak and modify files)")
    parser.add_argument("--verify-imports", action="store_true", help="(Optional) After applying, attempt to import modified modules to detect immediate import errors")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    modified = annotate(repo_root, apply_changes=args.apply)
    outdir = repo_root / "reports"
    outdir.mkdir(exist_ok=True)
    out = outdir / "actiondef_refs_fixed.txt"
    out.write_text("\n".join(modified), encoding="utf-8")
    print(f"Found {len(modified)} files referencing 'ActionDef'. Report: {out}")

    if args.verify_imports and args.apply and modified:
        # attempt to import each modified file as a module to detect immediate import errors
        import subprocess
        import sys
        failures = []
        for rel in modified:
            # convert path to module-like by replacing separators and stripping .py
            mod = rel.replace(str(repo_root) + os.sep, "")
            mod = rel[:-3].replace(os.sep, ".") if rel.endswith('.py') else rel
            try:
                subprocess.check_call([sys.executable, "-c", f"import importlib; importlib.import_module('{mod}')"], cwd=str(repo_root))
            except Exception:
                failures.append(rel)
        if failures:
            print("Import verification failed for:")
            for f in failures:
                print(" - ", f)
        else:
            print("Import verification: all modified modules imported successfully (no immediate ImportError)")


if __name__ == "__main__":
    main()
