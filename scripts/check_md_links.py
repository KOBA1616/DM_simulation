#!/usr/bin/env python3
"""リポジトリ内の Markdown ファイルを走査し、存在しない相対リンクを報告するスクリプト。

使い方: python scripts/check_md_links.py
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

IGNORED_DIRS = {"archive", "build-msvc", "third_party", "_deps", ".venv", "venv", "node_modules", "dist"}


def iter_md_files():
    for root, dirs, files in os.walk(ROOT):
        # prune ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for f in files:
            if f.lower().endswith(".md"):
                yield Path(root) / f


def normalize_link(link: str) -> str:
    # remove surrounding whitespace
    link = link.strip()
    # strip known schemes
    if link.startswith(("http://", "https://", "mailto:", "tel:", "//")):
        return "EXTERNAL"
    # strip anchor
    if "#" in link:
        link = link.split("#", 1)[0]
    return link


def resolve_link(md_path: Path, link: str) -> Path:
    link = normalize_link(link)
    if link == "" or link == "EXTERNAL":
        return None
    p = Path(link)
    if p.is_absolute():
        # treat as repo-root relative if starts with '/'
        candidate = (ROOT / p.relative_to(p.anchor)).resolve()
    else:
        candidate = (md_path.parent / p).resolve()
    return candidate


def main():
    broken = []
    total_checked = 0
    for md in iter_md_files():
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in MD_LINK_RE.finditer(text):
            link = m.group(1)
            norm = normalize_link(link)
            if norm == "EXTERNAL" or norm == "":
                continue
            total_checked += 1
            resolved = resolve_link(md, link)
            if resolved is None:
                continue
            # allow anchors-only links to existing file (resolved may be file.md)
            if not resolved.exists():
                broken.append((str(md.relative_to(ROOT)), link, str(resolved.relative_to(ROOT) if resolved.exists() else resolved)))

    if not broken:
        print("NO_BROKEN_LINKS")
        print(f"Checked {total_checked} relative links.")
        return 0

    print("BROKEN_LINKS:")
    for mdfile, link, resolved in broken:
        print(f"{mdfile}: {link} -> {resolved}")
    print(f"Total broken links: {len(broken)}")
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
