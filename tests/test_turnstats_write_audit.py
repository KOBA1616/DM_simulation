# -*- coding: utf-8 -*-
"""Audit contract test: detect direct writes to TurnStats protected fields.

Protected fields should only be updated via centralized helpers or StatCommand
implementation. This test fails if a direct assignment or '+=' to those fields
appears outside of allowed implementation files.
"""

import re
from pathlib import Path


PROTECTED_FIELDS = [
    "creatures_destroyed_this_turn",
    "summon_count_this_turn",
    "mana_set_this_turn",
    "shield_break_attempt_count_this_turn",
    "shield_break_resolved_count_this_turn",
]

# Files allowed to perform direct writes (helper implementations, StatCommand)
ALLOWED_PATHS = {
    "src/engine/infrastructure/commands/definitions/commands.cpp",
    "src/engine/infrastructure/commands/stat_update.hpp",
}


def find_violations():
    violations = []
    src_root = Path("src")
    pattern_template = r"turn_stats\.{field}\s*(?:\+=|=)"
    for file in src_root.rglob("*.cpp"):
        text = file.read_text(encoding="utf-8")
        for field in PROTECTED_FIELDS:
            pat = pattern_template.format(field=re.escape(field))
            if re.search(pat, text):
                rel = file.as_posix()
                if rel not in ALLOWED_PATHS:
                    violations.append((rel, field))

    # Also scan headers for accidental assignments
    for file in src_root.rglob("*.hpp"):
        text = file.read_text(encoding="utf-8")
        for field in PROTECTED_FIELDS:
            pat = pattern_template.format(field=re.escape(field))
            if re.search(pat, text):
                rel = file.as_posix()
                if rel not in ALLOWED_PATHS:
                    violations.append((rel, field))

    return violations


def test_no_unapproved_turnstats_writes():
    v = find_violations()
    assert not v, f"未承認の TurnStats 書き込みが見つかりました: {v}"
