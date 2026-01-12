# -*- coding: utf-8 -*-
"""
Shim entry point for the Editor.

Allows launching via `python -m dm_toolkit.gui.editor.main` by delegating to
the card editor's main.
"""
from typing import Optional, List
import sys

from dm_toolkit.gui.card_editor import main as card_editor_main


def main(argv: Optional[List[str]] = None) -> int:
    return card_editor_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
