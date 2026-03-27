import re

# Update zone_formatters.py
with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "r") as f:
    zf_content = f.read()

zf_content = zf_content.replace(
    "return TextUtils.apply_conjugation(text, optional)",
    "return text"
)
zf_content = zf_content.replace(
    "return LegacyActionFormatterHelper.apply_conjugation(command, text)",
    "return text"
)
zf_content = re.sub(r"optional = bool\(command\.get\(['\"]optional['\"], False\)\)\n\s+", "", zf_content)

with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "w") as f:
    f.write(zf_content)

# Update legacy_action_formatters.py
with open("dm_toolkit/gui/editor/formatters/legacy_action_formatters.py", "r") as f:
    laf_content = f.read()

laf_content = laf_content.replace(
    "return LegacyActionFormatterHelper.apply_conjugation(command, text)",
    "return text"
)
laf_content = laf_content.replace(
    "return LegacyActionFormatterHelper.apply_conjugation(action, text)",
    "return text"
)

with open("dm_toolkit/gui/editor/formatters/legacy_action_formatters.py", "w") as f:
    f.write(laf_content)

# Update draw_discard_formatters.py (if exists, checking contents first to be safe)
try:
    with open("dm_toolkit/gui/editor/formatters/draw_discard_formatters.py", "r") as f:
        dd_content = f.read()

    dd_content = dd_content.replace(
        "template = TextUtils.apply_conjugation(template, optional)",
        ""
    )
    dd_content = dd_content.replace(
        "return TextUtils.apply_conjugation(text, optional)",
        "return text"
    )
    dd_content = re.sub(r"optional = bool\(command\.get\(['\"]optional['\"], False\)\)\n\s+", "", dd_content)

    with open("dm_toolkit/gui/editor/formatters/draw_discard_formatters.py", "w") as f:
        f.write(dd_content)
except Exception as e:
    pass

# Update special_effect_formatters.py (AddKeyword, Mutate, SummonToken etc uses CommandFormatterBase directly or indirectly but doesn't have apply_conjugation)

# Update game_action_formatters.py (also uses optional?)
# No `apply_conjugation` in game_action_formatters.py based on grep, but let's check `SelectOptionFormatter` which uses optional internally to add `（同じものを選んでもよい）`. That's fine.

print("Removed individual optional conjugation calls.")
