import re
from typing import Tuple


CONNECT_RE = re.compile(r"(?P<obj>[\w\.]+)\.(?P<signal>[A-Za-z_]\w*)\.connect\(")


def replace_connects_in_text(text: str) -> Tuple[str, int]:
    """Replace occurrences of `obj.signal.connect(` with
    `safe_connect(obj, 'signal', ` and return (new_text, replacements_count).
    This is intentionally conservative and only handles simple dotted-name
    objects (e.g., `btn.clicked.connect(`, `self.timer.timeout.connect(`).
    """

    def _repl(m: re.Match) -> str:
        obj = m.group('obj')
        sig = m.group('signal')
        return f"safe_connect({obj}, '{sig}', "

    new_text, n = CONNECT_RE.subn(_repl, text)
    return new_text, n


def ensure_safe_connect_import(text: str) -> Tuple[str, bool]:
    """Ensure the import `from dm_toolkit.gui.editor.forms.signal_utils import safe_connect`
    exists in the file. If added, return (new_text, True). If already present, return (text, False).
    The import is inserted after the last existing import statement.
    """
    import_token = 'from dm_toolkit.gui.editor.forms.signal_utils import safe_connect'
    if import_token in text:
        return text, False

    import_stmt = 'from dm_toolkit.gui.editor.forms.signal_utils import safe_connect\n'

    # find the last import line
    lines = text.splitlines(keepends=True)
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i

    if last_import_idx >= 0:
        insert_at = last_import_idx + 1
        lines.insert(insert_at, import_stmt)
    else:
        # no imports; place at top
        lines.insert(0, import_stmt)

    return ''.join(lines), True


def process_file(path, apply: bool = False) -> Tuple[int, bool]:
    """Process a file: compute replacements, optionally apply them.
    Returns (replacements_count, import_added).
    """
    p = path
    text = p.read_text(encoding='utf-8')
    new_text, n = replace_connects_in_text(text)
    import_added = False
    if n > 0:
        new_text, added = ensure_safe_connect_import(new_text)
        import_added = added
        if apply:
            p.write_text(new_text, encoding='utf-8')
    return n, import_added


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description='Auto-replace .connect with safe_connect')
    ap.add_argument('root', nargs='?', default='.', help='Project root')
    ap.add_argument('--apply', action='store_true', help='Apply changes')
    args = ap.parse_args()

    root = Path(args.root)
    total = 0
    files_changed = 0
    for p in root.rglob('*.py'):
        # skip tests
        if 'python/tests' in p.as_posix():
            continue
        n, added = process_file(p, apply=args.apply)
        if n > 0:
            print(f"{p}: replacements={n}, import_added={added}")
            total += n
            files_changed += 1

    print(f"Total replacements: {total} in {files_changed} files")
