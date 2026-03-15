#!/usr/bin/env python3
"""Find command types listed in consts.COMMAND_TYPES that are not mapped
in CommandModel.ingest_legacy_structure for typed params.
"""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
models_py = ROOT / 'dm_toolkit' / 'gui' / 'editor' / 'models' / '__init__.py'
consts_py = ROOT / 'dm_toolkit' / 'consts.py'

# Extract mapped cmd_types from models.__init__.py
src = models_py.read_text(encoding='utf-8')
# find occurrences like: if cmd_type == 'QUERY':
mapped = set(re.findall(r"elif cmd_type == '([A-Z0-9_]+)'|if cmd_type == '([A-Z0-9_]+)'", src))
mapped_types = set()
for a,b in mapped:
    mapped_types.add(a or b)

# Extract registered command names from schema_config.py (register_schema(CommandSchema("NAME", ...))
schema_src = (ROOT / 'dm_toolkit' / 'gui' / 'editor' / 'schema_config.py').read_text(encoding='utf-8')
regs = re.findall(r"register_schema\(CommandSchema\(\s*\"([A-Z0-9_]+)\"", schema_src)
registered = regs

unmapped = [c for c in registered if c not in mapped_types]
print('Mapped types in ingest_legacy_structure:', sorted(mapped_types))
print('\nTotal registered command schemas:', len(registered))
print('Registered commands missing typed params mapping:')
for c in unmapped:
    print(' -', c)

# Output summary file
out = ROOT / 'scripts' / 'untyped_commands.json'
import json
with open(out, 'w', encoding='utf-8') as f:
    json.dump({'mapped': sorted(list(mapped_types)), 'registered': registered, 'unmapped': unmapped}, f, ensure_ascii=False, indent=2)
print('\nWritten', out)
