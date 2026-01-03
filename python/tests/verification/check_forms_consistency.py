import re
import pathlib

root = pathlib.Path('c:/Users/ichirou/DM_simulation/dm_toolkit/gui/editor/forms')
files = list(root.glob('*.py'))

report = {}
attr_use_re = re.compile(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)")
assign_re = re.compile(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=")

for f in files:
    text = f.read_text(encoding='utf-8')
    uses = set(attr_use_re.findall(text))
    assigns = set(assign_re.findall(text))
    # Attributes used but not assigned in same file
    missing = sorted(list(uses - assigns))
    # Filter out common cases where widget may be provided by super class: known_shared = {...}
    report[f.name] = {'used': sorted(uses), 'assigned': sorted(assigns), 'missing': missing}

import json
print(json.dumps(report, indent=2, ensure_ascii=False))
