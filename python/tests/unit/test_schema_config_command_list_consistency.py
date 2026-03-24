from pathlib import Path
import re
import json


ROOT = Path(__file__).resolve().parents[3]
schema_py = ROOT / 'dm_toolkit' / 'gui' / 'editor' / 'schema_config.py'
expected_file = ROOT / 'data' / 'expected_registered_commands.json'


def extract_registered_cmds(src: str):
    return re.findall(r"register_schema\(CommandSchema\(\s*\"([A-Z0-9_]+)\"", src)


def test_registered_commands_match_expected():
    src = schema_py.read_text(encoding='utf-8')
    registered = extract_registered_cmds(src)

    expected = json.loads(expected_file.read_text(encoding='utf-8'))

    assert registered == expected, f"Registered commands differ from expected baseline.\n\nRegistered: {registered}\nExpected:   {expected}"
