#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Minimal CLI tool to simulate player selecting targets for pending effects.
# Usage: python tools/cli_target_selector.py <game_state_dump.json>
# The game_state_dump.json should be a JSON export produced by the engine containing pending_effects and zones with instance ids.

if len(sys.argv) < 2:
    print("Usage: python tools/cli_target_selector.py <game_state_dump.json>")
    sys.exit(1)

p = Path(sys.argv[1])
if not p.exists():
    print("File not found:", p)
    sys.exit(1)

with p.open('r', encoding='utf-8') as f:
    gs = json.load(f)

pending = gs.get('pending_effects', [])
players = gs.get('players', [])

if not pending:
    print('No pending effects')
    sys.exit(0)

print('Pending Effects')
for idx, pe in enumerate(pending):
    print(f"[{idx}] type={pe.get('type')} src={pe.get('source_instance_id')} controller={pe.get('controller')} num_needed={pe.get('num_targets_needed')}")

sel = int(input('Select pending effect index: '))
pe = pending[sel]
need = pe.get('num_targets_needed', 0)
print('Need targets:', need)

# list all battle zone instances
instances = []
for pl_idx, pl in enumerate(players):
    for c in pl.get('battle_zone', []):
        instances.append({'player': pl_idx, 'instance_id': c.get('instance_id'), 'card_id': c.get('card_id')})

print('Available instances:')
for i, it in enumerate(instances):
    print(f"[{i}] player={it['player']} instance={it['instance_id']} card_id={it['card_id']}")

chosen = []
for i in range(need):
    idx = int(input(f"Choose instance index for target #{i+1}: "))
    chosen.append(instances[idx]['instance_id'])

print('Selected targets:', chosen)

# Create an output action sequence that can be consumed by engine (format is simple JSON list of SELECT_TARGET actions)
actions = []
slot_index = sel
for t in chosen:
    actions.append({
        'type': 'SELECT_TARGET',
        'slot_index': slot_index,
        'target_instance_id': t
    })

outp = Path('tools/selected_targets.json')
with outp.open('w', encoding='utf-8') as f:
    json.dump(actions, f, indent=2)

print('Wrote', outp)
