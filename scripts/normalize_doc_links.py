#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

EXTS = [".md", ".py", ".rst", ".txt"]

dup_re = re.compile(r"(/|\\b)(?P<name>[^/()\\\s]+)/(?P=name)(/|\\b)")

def should_process(path: Path) -> bool:
    return path.suffix in EXTS or path.name.lower().startswith("readme")

def normalize(text: str) -> str:
    # Iteratively collapse immediate duplicated path segments like 'Specs' -> 'Specs'
    prev = None
    while prev != text:
        prev = text
        # replace occurrences like '/Specs/' or 'Specs/' or '/Specs'
        text = re.sub(r"(?P<prefix>/|\b)(?P<name>[^/()\\\s]+)/(?P=name)(?P<suffix>/|\b)", lambda m: f"{m.group('prefix')}{m.group('name')}{m.group('suffix')}", text)
    return text

def main():
    changed = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if not should_process(p):
            continue
        try:
            txt = p.read_text(encoding='utf-8')
        except Exception:
            continue
        new = normalize(txt)
        if new != txt:
            p.write_text(new, encoding='utf-8')
            changed.append(str(p.relative_to(ROOT)))
    print("NORMALIZED:\n" + "\n".join(changed))

if __name__ == '__main__':
    main()
