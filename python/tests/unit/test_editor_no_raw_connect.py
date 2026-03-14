import re
from pathlib import Path


def test_no_raw_connect_in_editor():
    root = Path("dm_toolkit/gui/editor")
    assert root.exists(), f"{root} not found"
    pattern = re.compile(r"\\.connect\s*\(")
    matches = []
    for p in root.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if pattern.search(text):
            matches.append(str(p))
    assert not matches, "Found raw .connect() in files:\n" + "\n".join(matches)
