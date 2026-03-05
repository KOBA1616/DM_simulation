import json
import sys
import pathlib
# Ensure repo root is on sys.path so dm_ai_module can be imported
repo_root = str(pathlib.Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import dm_ai_module as dm
from dm_toolkit.command_builders import (
    build_pass_command,
    build_mana_charge_command,
    build_play_card_command,
    build_attack_player_command
)

def show(cmd_dict):
    try:
        if isinstance(cmd_dict, dict):
            # Convert dict to CommandDef
            cmd = dm.CommandDef()
            if 'type' in cmd_dict:
                t = cmd_dict['type']
                if isinstance(t, str):
                    try:
                        cmd.type = getattr(dm.CommandType, t)
                    except:
                        if t == 'PLAY': cmd.type = dm.CommandType.PLAY_FROM_ZONE
                        else: pass # NONE
            if 'slot_index' in cmd_dict: cmd.slot_index = cmd_dict['slot_index']
            if 'instance_id' in cmd_dict: cmd.instance_id = cmd_dict['instance_id']
        else:
            cmd = cmd_dict

        idx = dm.CommandEncoder.command_to_index(cmd)
    except Exception as e:
        idx = f"ERROR: {e}"
    print(json.dumps({'cmd': cmd_dict, 'index': idx}, ensure_ascii=False))

# Construct commands using builders where possible, falling back to dicts for edge cases or non-supported fields
samples = [
    build_pass_command(native=False),
    build_mana_charge_command(source_instance_id=0, slot_index=3, native=False), # instance_id 0 dummy
    build_mana_charge_command(source_instance_id=0, native=False),
    build_play_card_command(card_id=0, source_instance_id=0, slot_index=5, native=False),
    {'type': 'PLAY', 'slot_index': 7}, # Legacy type explicit test
    build_play_card_command(card_id=0, source_instance_id=0, native=False),
    build_attack_player_command(attacker_instance_id=42, target_player=0, native=False),
    {'type': 'ATTACK'}, # Legacy/Partial
    {'type': 'UNKNOWN', 'foo': 'bar'},
]

for c in samples:
    show(c)
