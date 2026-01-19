#!/usr/bin/env python3
import re
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "archive" / "docs_link_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: old basename -> new repo-relative path
MAPPING = {
    "01_Game_Engine_Specs.md": "docs/Specs/01_Game_Engine_Specs.md",
    "01_System_Architecture_Overview_JP.md": "docs/Specs/01_System_Architecture_Overview_JP.md",
    "02_AI_System_Specs.md": "docs/Specs/02_AI_System_Specs.md",
    "03_Card_Editor_Specs.md": "docs/Specs/03_Card_Editor_Specs.md",
    "AGENTS.md": "docs/Specs/AGENTS.md",
    "CARD_MOVEMENT_OUTPUT.md": "docs/spell/CARD_MOVEMENT_OUTPUT.md",
    "DISCARD_OUTPUT.md": "docs/spell/DISCARD_OUTPUT.md",
    "REPLACE_CARD_MOVE_USAGE.md": "docs/guides/REPLACE_CARD_MOVE_USAGE.md",
    "GUI_HEADLESS_TESTING_SETUP.md": "docs/guides/GUI_HEADLESS_TESTING_SETUP.md",
    "GUI_VARIABLE_LINK_FIX.md": "docs/guides/GUI_VARIABLE_LINK_FIX.md",
    "shield_break_buffer_design.md": "docs/design/shield_break_buffer_design.md",
    "MEGA_LAST_BURST_IMPLEMENTATION.md": "archive/docs/archive/docs/archive/docs/archive/docs/archive/docs/MEGA_LAST_BURST_IMPLEMENTATION.md",
    "IF_CONDITION_LABELS.md": "archive/docs/archive/docs/archive/docs/archive/docs/archive/docs/IF_CONDITION_LABELS.md",
    "directory_cleanup_report_20260119.md": "docs/reports/directory_cleanup_report_20260119.md",
    "REPO_LFS_MIGRATION_NOTICE.md": "docs/reports/REPO_LFS_MIGRATION_NOTICE.md",
    "telemetry_run.md": "docs/reports/telemetry_run.md",
    "requirements_review_20260119.md": "docs/requirements/requirements_review_20260119.md",
    "Research_Presentation_JP.md": "docs/results/Research_Presentation_JP.md",
    "action_command_mapping.md": "docs/api/action_command_mapping.md",
}

EXTS = [".md", ".py", ".rst", ".txt"]

def backup_file(path: Path):
    rel = path.relative_to(ROOT)
    dest = BACKUP_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)

def replace_in_text(text: str, name: str, newpath: str) -> str:
    # do several contextual replacements to preserve relative forms
    # pattern examples: docs/NAME, ../NAME, ./NAME, (NAME, /NAME
    # 1) docs/NAME -> newpath
    text = text.replace(f"docs/{name}", newpath)
    # 2) ../NAME -> ../ + (newpath without leading docs/)
    if newpath.startswith("docs/"):
        suffix = newpath[len("docs/"):]
    else:
        suffix = newpath
    text = text.replace(f"../{name}", f"../{suffix}")
    text = text.replace(f"./{name}", f"./{suffix}")
    text = text.replace(f"({name}", f"({suffix}")
    text = text.replace(f"/{name}", f"/{suffix}")
    return text

def should_process(path: Path) -> bool:
    return path.suffix in EXTS or path.name.lower().startswith("readme")

def main():
    files = list(ROOT.rglob("*"))
    changed = []
    for p in files:
        if not p.is_file():
            continue
        if not should_process(p):
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        original = txt
        for name, new in MAPPING.items():
            txt = replace_in_text(txt, name, new)
        if txt != original:
            backup_file(p)
            p.write_text(txt, encoding="utf-8")
            changed.append(str(p.relative_to(ROOT)))

    print("UPDATED_FILES:\n" + "\n".join(changed))

if __name__ == "__main__":
    main()
