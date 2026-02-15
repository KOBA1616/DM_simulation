import json
import sys
import pathlib
# Ensure repo root is on sys.path so dm_ai_module can be imported
repo_root = str(pathlib.Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import dm_ai_module as dm

def show(cmd):
    try:
        idx = dm.CommandEncoder.command_to_index(cmd)
    except Exception as e:
        idx = f"ERROR: {e}"
    print(json.dumps({'cmd': cmd, 'index': idx}, ensure_ascii=False))

samples = [
    {'type': 'PASS'},
    {'type': 'MANA_CHARGE', 'slot_index': 3},
    {'type': 'MANA_CHARGE'},
    {'type': 'PLAY_FROM_ZONE', 'slot_index': 5},
    {'type': 'PLAY', 'slot_index': 7},
    {'type': 'PLAY_FROM_ZONE'},
    {'type': 'ATTACK', 'instance_id': 42},
    {'type': 'ATTACK'},
    {'type': 'UNKNOWN', 'foo': 'bar'},
]

for c in samples:
    show(c)
