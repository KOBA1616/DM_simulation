import re
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[3]
models_py = ROOT / 'dm_toolkit' / 'gui' / 'editor' / 'models' / '__init__.py'
schema_py = ROOT / 'dm_toolkit' / 'gui' / 'editor' / 'schema_config.py'


def extract_mapped_cmds(src: str):
    pairs = re.findall(r"elif cmd_type == '([A-Z0-9_]+)'|if cmd_type == '([A-Z0-9_]+)'", src)
    mapped = set()
    for a, b in pairs:
        mapped.add(a or b)
    return mapped


def extract_registered_cmds(src: str):
    return re.findall(r"register_schema\(CommandSchema\(\s*\"([A-Z0-9_]+)\"", src)


def test_high_priority_registered_commands_have_typed_params():
    """
    Ensure a small set of high-priority command schemas are mapped
    in `models.__init__.py` ingest logic to typed params.
    """
    models_src = models_py.read_text(encoding='utf-8')
    schema_src = schema_py.read_text(encoding='utf-8')

    mapped = extract_mapped_cmds(models_src)
    # High-priority commands expected to be typed (subset)
    expected_mapped = {
        'PUT_CREATURE', 'REPLACE_CARD_MOVE', 'SEND_SHIELD_TO_GRAVE', 'LOOK_TO_BUFFER',
        'SUMMON_TOKEN', 'POWER_MOD', 'ADD_KEYWORD', 'REVEAL_CARDS', 'COUNT_CARDS',
        'DRAW_CARD', 'DISCARD', 'MOVE_CARD'
    }

    missing = sorted([c for c in expected_mapped if c not in mapped])
    assert missing == [], f"High-priority commands missing typed params mapping: {missing}"
